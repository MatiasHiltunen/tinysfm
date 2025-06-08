use anyhow::{Context, Result};
use base64::prelude::*;
use brotli::BrotliDecompress;
use clap::Parser;
use flate2::read::{DeflateDecoder, GzDecoder};
use futures::stream::StreamExt;
use governor::{Quota, RateLimiter};
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use std::num::NonZeroU32;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration as TokioDuration};

#[derive(Parser, Debug)]
#[clap(about = "Advanced large file downloader")]
struct Cli {
    #[arg(short, long, help = "URL to download")]
    url: String,

    #[arg(short, long, default_value = "downloaded_file", help = "Output file path")]
    output: String,

    #[arg(short = 'n', long, default_value_t = 4, help = "Number of parallel downloads")]
    parallel: usize,

    #[arg(long, default_value_t = 10_485_760, help = "Chunk size in bytes (default: 10MB)")]
    chunk_size: u64,

    #[arg(long, default_value_t = 3, help = "Max retries per chunk")]
    retries: u32,

    #[arg(long, help = "Expected MD5 checksum for verification")]
    checksum: Option<String>,

    #[arg(long, help = "Rate limit in bytes per second (e.g., 1048576 for 1MB/s)")]
    rate_limit: Option<u32>,

    #[arg(
        long,
        default_value = "gzip, deflate, br, zstd",
        help = "Accept-Encoding header value"
    )]
    accept_encoding: String,

    #[arg(long, help = "Username for basic auth")]
    username: Option<String>,

    #[arg(long, help = "Password for basic auth")]
    password: Option<String>,

    #[arg(long, help = "Bearer token for authentication")]
    token: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct DownloadState {
    completed_chunks: Vec<(u64, u64)>,
    file_size: u64,
}

#[derive(Clone)]
struct Downloader {
    client: Client,
    url: String,
    output_path: String,
    chunk_size: u64,
    retries: u32,
    checksum: Option<String>,
    rate_limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    accept_encoding: String,
    auth_header: Option<String>,
}

impl Downloader {
    fn new(args: Cli) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        let auth_header = if let (Some(username), Some(password)) = (&args.username, &args.password) {
            let auth = format!("Basic {}", BASE64_STANDARD.encode(format!("{}:{}", username, password)));
            headers.insert("Authorization", auth.parse().unwrap());
            Some(auth)
        } else if let Some(token) = &args.token {
            let auth = format!("Bearer {}", token);
            headers.insert("Authorization", auth.parse().unwrap());
            Some(auth)
        } else {
            None
        };
        headers.insert("Accept-Encoding", args.accept_encoding.parse().unwrap());

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client")?;

        let rate_limiter = args.rate_limit.map(|limit| {
            let limit = NonZeroU32::new(limit).unwrap_or_else(|| NonZeroU32::new(1).unwrap());
            Arc::new(RateLimiter::direct(Quota::per_second(limit)))
        });

        Ok(Downloader {
            client,
            url: args.url,
            output_path: args.output,
            chunk_size: args.chunk_size,
            retries: args.retries,
            checksum: args.checksum,
            rate_limiter,
            accept_encoding: args.accept_encoding,
            auth_header,
        })
    }

    async fn get_file_size(&self) -> Result<u64> {
        let resp = self
            .client
            .head(&self.url)
            .send()
            .await
            .context("Failed to send HEAD request")?;
        resp.headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .context("Failed to get content length")
    }

    async fn check_range_support(&self) -> bool {
        let resp = self.client.head(&self.url).send().await.ok();
        resp.map(|r| r.headers().get("accept-ranges").map(|v| v == "bytes").unwrap_or(false))
            .unwrap_or(false)
    }

    async fn try_download_chunk(&self, start: u64, end: u64, attempt: u32) -> Result<Vec<u8>> {
        if let Some(limiter) = &self.rate_limiter {
            limiter.until_ready().await;
        }
        let range = format!("bytes={}-{}", start, end);
        let resp = self
            .client
            .get(&self.url)
            .header("Range", &range)
            .send()
            .await
            .context("Failed to send chunk request")?;
        let content_encoding = resp
            .headers()
            .get("content-encoding")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let bytes = resp.bytes().await.context("Failed to read chunk bytes")?;

        if let Some(encoding) = content_encoding.as_deref() {
            if encoding != "gzip" && encoding != "deflate" && encoding != "br" {
                match encoding {
                    "zstd" => zstd::stream::decode_all(&bytes[..])
                        .map_err(|e| anyhow::anyhow!("Zstd decompression error: {}", e)),
                    _ => Err(anyhow::anyhow!("Unsupported content encoding: {}", encoding)),
                }
            } else {
                Ok(bytes.to_vec())
            }
        } else {
            Ok(bytes.to_vec())
        }
    }

    async fn save_state(&self, completed_chunks: &[(u64, u64)]) -> Result<()> {
        let state = DownloadState {
            completed_chunks: completed_chunks.to_vec(),
            file_size: self.get_file_size().await?,
        };
        let state_path = format!("{}.state", self.output_path);
        let json = serde_json::to_string(&state)?;
        fs::write(&state_path, json)?;
        Ok(())
    }

    async fn load_state(&self) -> Result<Option<DownloadState>> {
        let state_path = format!("{}.state", self.output_path);
        if Path::new(&state_path).exists() {
            let json = fs::read_to_string(&state_path)?;
            let state = serde_json::from_str(&json)?;
            Ok(Some(state))
        } else {
            Ok(None)
        }
    }

    async fn download_chunk(&self, start: u64, end: u64) -> Result<Vec<u8>> {
        let mut attempt = 0;
        loop {
            match self.try_download_chunk(start, end, attempt).await {
                Ok(bytes) => return Ok(bytes),
                Err(e) if attempt < self.retries => {
                    attempt += 1;
                    let delay = TokioDuration::from_millis(100 * 2u64.pow(attempt) + (rand::random::<u64>() % 100));
                    warn!(
                        "Chunk {}-{} failed (attempt {}/{}) with {}. Retrying in {:?}",
                        start,
                        end,
                        attempt,
                        self.retries,
                        e,
                        delay
                    );
                    sleep(delay).await;
                }
                Err(e) => {
                    return Err(e.context(format!(
                        "Failed to download chunk {}-{} after {} retries",
                        start, end, self.retries
                    )));
                }
            }
        }
    }

    async fn download(&self) -> Result<()> {
        let file_size = self.get_file_size().await?;
        let supports_ranges = self.check_range_support().await;
        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Downloading");

        if !supports_ranges {
            warn!("Server does not support range requests. Downloading in a single thread.");
            let resp = self.client.get(&self.url).send().await?;
            let mut file = File::create(&self.output_path)?;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk)?;
                pb.inc(chunk.len() as u64);
            }
            pb.finish_with_message("Download complete");
            return Ok(());
        }

        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.output_path)?;
        file.set_len(file_size)?;

        let state = self.load_state().await?;
        let mut completed_chunks = state
            .map(|s| s.completed_chunks.into_iter().collect::<HashSet<_>>())
            .unwrap_or_default();

        let (state_tx, mut state_rx) = mpsc::channel::<(u64, u64)>(100);
        let downloader = self.clone();
        let state_saver = tokio::spawn(async move {
            let mut chunks = Vec::new();
            while let Some((start, end)) = state_rx.recv().await {
                chunks.push((start, end));
                if chunks.len() % 10 == 0 {
                    if let Err(e) = downloader.save_state(&chunks).await {
                        error!("Failed to save state: {}", e);
                    }
                }
            }
            if let Err(e) = downloader.save_state(&chunks).await {
                error!("Failed to save state: {}", e);
            }
        });

        let mut tasks = Vec::new();
        for start in (0..file_size).step_by(self.chunk_size as usize) {
            let end = (start + self.chunk_size - 1).min(file_size - 1);
            if completed_chunks.contains(&(start, end)) {
                pb.inc(end - start + 1);
                continue;
            }

            let downloader = self.clone();
            let state_tx = state_tx.clone();
            let pb = pb.clone();
            tasks.push(tokio::spawn(async move {
                let bytes = downloader.download_chunk(start, end).await?;
                let mut file = File::options().write(true).open(&downloader.output_path)?;
                file.seek(SeekFrom::Start(start))?;
                file.write_all(&bytes)?;
                pb.inc(end - start + 1);
                state_tx.send((start, end)).await?;
                Ok::<(), anyhow::Error>(())
            }));
        }

        for task in tasks {
            task.await??;
        }

        drop(state_tx);
        state_saver.await?;

        pb.finish_with_message("Download complete");

        if let Some(expected_checksum) = &self.checksum {
            let mut file = File::open(&self.output_path)?;
            let mut hasher = md5::Context::new();
            let mut buffer = vec![0u8; 8192];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hasher.consume(&buffer[..n]);
            }
            let computed = format!("{:x}", hasher.compute());
            if computed != *expected_checksum {
                return Err(anyhow::anyhow!(
                    "Checksum mismatch: expected {}, got {}",
                    expected_checksum,
                    computed
                ));
            }
            info!("Checksum verified successfully");
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();
    let args = Cli::parse();

    let downloader = Downloader::new(args)?;
    let downloader_clone = downloader.clone();

    tokio::spawn(async move {
        signal::ctrl_c().await.unwrap();
        info!("Received Ctrl+C, saving state...");
        if let Err(e) = downloader_clone.save_state(&[]).await {
            error!("Failed to save state on exit: {}", e);
        }
        std::process::exit(0);
    });

    downloader.download().await?;
    Ok(())
}

fn decompress(data: &[u8], encoding: Option<&str>) -> Result<Vec<u8>> {
    match encoding {
        Some("br") => {
            let mut decompressed = Vec::new();
            BrotliDecompress(&mut io::Cursor::new(data), &mut decompressed)?;
            Ok(decompressed)
        }
        Some("gzip") => {
            let mut decompressed = Vec::new();
            let mut decoder = GzDecoder::new(data);
            decoder.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        }
        Some("deflate") => {
            let mut decompressed = Vec::new();
            let mut decoder = DeflateDecoder::new(data);
            decoder.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        }
        Some("zstd") => zstd::stream::decode_all(data)
            .map_err(|e| anyhow::anyhow!("Zstd decompression error: {}", e)),
        _ => Ok(data.to_vec()),
    }
}
