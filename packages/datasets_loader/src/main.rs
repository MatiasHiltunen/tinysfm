use anyhow::{Context, Result};
use futures::{stream, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::{header, Client};
use serde::Deserialize;
use std::{path::{Path, PathBuf}, sync::Arc, time::Duration};
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncSeekExt, AsyncWriteExt},
    sync::Semaphore,
    time::timeout,
};
use tokio_retry::{strategy::ExponentialBackoff, Retry};
use zip::ZipArchive;

#[derive(Debug, Deserialize)]
struct Dataset {
    name: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    datasets: Vec<Dataset>,
}

const TARGET_DIR: &str = "../../tests/data";
const MAX_CONCURRENT_DOWNLOADS: usize = 3;
const MAX_CHUNKS_PER_FILE: usize = 8;
const MIN_CHUNK_SIZE: u64 = 8 * 1024 * 1024; // 8 MiB minimum chunk
const MAX_CHUNK_SIZE: u64 = 16 * 1024 * 1024; // 16 MiB maximum chunk
const CHUNK_TIMEOUT: Duration = Duration::from_secs(60); // 60 seconds per chunk
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30); // 30 seconds for initial request

/// Entry point: parse manifest, download, and extract.
pub async fn process_manifest(manifest_path: &Path) -> Result<()> {
    let data = fs::read_to_string(manifest_path)
        .await
        .with_context(|| format!("Couldn't read manifest at {manifest_path:?}"))?;
    let manifest: Manifest = serde_json::from_str(&data)?;

    fs::create_dir_all(TARGET_DIR).await?;

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(REQUEST_TIMEOUT)
        .tcp_keepalive(Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .pool_max_idle_per_host(10)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_DOWNLOADS));
    let multi = MultiProgress::new();

    // Spawn download tasks
    let handles: Vec<_> = manifest.datasets.into_iter().map(|ds| {
        let client = client.clone();
        let semaphore = semaphore.clone();
        let pb = multi.add(ProgressBar::new_spinner());
        
        tokio::spawn(async move {
            let _permit = semaphore.acquire().await?;
            let result = download_and_extract(&client, &ds, &pb).await;
            if let Err(ref e) = result {
                pb.finish_with_message(format!("❌ Failed: {}", e));
            }
            result
        })
    }).collect();

    // Wait for all downloads
    let mut failed = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => {},
            Ok(Err(e)) => {
                eprintln!("Download failed: {}", e);
                failed += 1;
            },
            Err(e) => {
                eprintln!("Task failed: {}", e);
                failed += 1;
            }
        }
    }

    if failed > 0 {
        println!("⚠️  {} downloads failed", failed);
    } else {
        println!("✅ All downloads completed successfully!");
    }
    
    Ok(())
}

async fn download_and_extract(client: &Client, ds: &Dataset, pb: &ProgressBar) -> Result<()> {
    pb.set_prefix(format!("📥 {}", ds.name));
    pb.enable_steady_tick(Duration::from_millis(200));
    pb.set_message("Getting file info...");

    // Get file size with retries
    let total = match get_content_length_with_retry(client, &ds.url).await {
        Ok(size) => size,
        Err(e) => {
            pb.finish_with_message(format!("❌ Failed to get size: {}", e));
            return Err(e);
        }
    };
    
    let out_path = target_path(&ds.url);
    
    // Check if already extracted
    let extracted_marker = get_extracted_marker_path(&out_path);
    if extracted_marker.exists() {
        pb.finish_with_message("✅ Already extracted");
        return Ok(());
    }
    
    // For multi-part archives, check if this is a part file
    let is_part_file = out_path.to_string_lossy().contains(".z0") && 
                       !out_path.to_string_lossy().ends_with(".zip");
    
    // Check if file already exists with correct size
    if let Ok(metadata) = fs::metadata(&out_path).await {
        if metadata.len() == total {
            if is_part_file {
                // For part files, just mark as downloaded
                pb.finish_with_message("✅ Part file downloaded");
                return Ok(());
            }
            
            pb.set_message("Found existing download, extracting...");
            // Extract the existing file
            if out_path.extension().and_then(|s| s.to_str()) == Some("zip") {
                match extract_zip(&out_path, TARGET_DIR).await {
                    Ok(_) => {
                        // Mark as extracted
                        fs::write(&extracted_marker, "extracted").await?;
                        // Clean up zip file
                        let _ = fs::remove_file(&out_path).await;
                        pb.finish_with_message("✅ Extracted existing file");
                    }
                    Err(e) => {
                        pb.finish_with_message(format!("❌ Extraction failed: {}", e));
                        return Err(e);
                    }
                }
            }
            return Ok(());
        }
    }
    
    // Calculate optimal chunk configuration
    let chunk_config = calculate_chunk_config(total);
    
    pb.set_length(total);
    pb.set_style(progress_style());
    pb.set_message(format!("{} chunks", chunk_config.num_chunks));

    // Download with advanced parallel chunking
    if let Err(e) = download_parallel_chunks(client, &ds.url, &out_path, total, &chunk_config, pb).await {
        pb.finish_with_message(format!("❌ Download failed: {}", e));
        return Err(e);
    }
    
    // For part files, we're done after downloading
    if is_part_file {
        pb.finish_with_message("✅ Part file complete");
        return Ok(());
    }
    
    pb.set_message("Processing...");
    
    // Handle multi-part archives (only for .zip files that might have parts)
    let final_path = if should_check_multipart(&out_path) {
        match assemble_multipart(&out_path).await {
            Ok(path) => {
                pb.set_message("Assembled multi-part archive");
                path
            }
            Err(_) => {
                // If assembly fails, it might be a single file
                out_path.clone()
            }
        }
    } else {
        out_path.clone()
    };
    
    // Extract zip files
    if final_path.extension().and_then(|s| s.to_str()) == Some("zip") {
        pb.set_message("Extracting archive...");
        match extract_zip(&final_path, TARGET_DIR).await {
            Ok(_) => {
                // Mark as extracted
                fs::write(&extracted_marker, "extracted").await?;
                
                // Clean up zip file after extraction
                pb.set_message("Cleaning up...");
                let _ = fs::remove_file(&final_path).await;
                
                pb.finish_with_message("✅ Complete");
            }
            Err(e) => {
                pb.finish_with_message(format!("❌ Extraction failed: {}", e));
                return Err(e);
            }
        }
    } else {
        pb.finish_with_message("✅ Downloaded");
    }
    
    Ok(())
}

#[derive(Debug)]
struct ChunkConfig {
    num_chunks: usize,
    concurrent_chunks: usize,
}

fn calculate_chunk_config(total_size: u64) -> ChunkConfig {
    // Conservative chunk sizing for stability
    let num_chunks = if total_size < 10 * 1024 * 1024 {
        // Very small files (<10MB): single chunk
        1
    } else if total_size < 100 * 1024 * 1024 {
        // Small files (10-100MB): 2-4 chunks
        std::cmp::min(4, std::cmp::max(2, (total_size / MIN_CHUNK_SIZE) as usize))
    } else {
        // Large files (>100MB): use MAX_CHUNK_SIZE
        let ideal_chunks = (total_size / MAX_CHUNK_SIZE) + 1;
        std::cmp::min(MAX_CHUNKS_PER_FILE, ideal_chunks as usize)
    };
    
    // Conservative concurrent chunks
    let concurrent_chunks = std::cmp::min(4, num_chunks);
    
    ChunkConfig {
        num_chunks,
        concurrent_chunks,
    }
}

async fn get_content_length_with_retry(client: &Client, url: &str) -> Result<u64> {
    let retry_strategy = ExponentialBackoff::from_millis(1000)
        .factor(2)
        .take(3);
    
    Retry::spawn(retry_strategy, || async {
        get_content_length(client, url).await
    }).await
}

async fn get_content_length(client: &Client, url: &str) -> Result<u64> {
    // Try HEAD request first
    match timeout(Duration::from_secs(10), client.head(url).send()).await {
        Ok(Ok(response)) => {
            if let Some(len) = response
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|h| h.to_str().ok()?.parse().ok())
            {
                return Ok(len);
            }
        }
        _ => {}
    }
    
    // Fallback: use Range request
    let response = timeout(
        Duration::from_secs(10),
        client
            .get(url)
            .header(header::RANGE, "bytes=0-0")
            .send()
    ).await
    .context("Timeout getting content length")??;
    
    response
        .headers()
        .get(header::CONTENT_RANGE)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split('/').last())
        .and_then(|n| n.parse().ok())
        .context("Server doesn't support partial content or didn't provide content length")
}

fn target_path(url: &str) -> PathBuf {
    let filename = url.rsplit('/').next().unwrap_or("download.bin");
    Path::new(TARGET_DIR).join(filename)
}

async fn download_parallel_chunks(
    client: &Client,
    url: &str,
    dest: &Path,
    total: u64,
    config: &ChunkConfig,
    pb: &ProgressBar,
) -> Result<()> {
    // Pre-allocate file
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await?;
    }
    
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dest)
        .await?;
    file.set_len(total).await?;
    file.sync_all().await?;
    drop(file);
    
    // Calculate chunk ranges - ensure no overlaps and complete coverage
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let chunk_size = total / config.num_chunks as u64;
    
    for i in 0..config.num_chunks {
        let start = i as u64 * chunk_size;
        let end = if i == config.num_chunks - 1 {
            // Last chunk gets all remaining bytes
            total - 1
        } else {
            // Non-overlapping ranges: [start, end] inclusive
            start + chunk_size - 1
        };
        
        ranges.push((start, end));
    }
    
    pb.set_message(format!("Downloading {} chunks ({} bytes each)", config.num_chunks, chunk_size));
    
    // Download chunks in parallel with controlled concurrency
    let semaphore = Arc::new(Semaphore::new(config.concurrent_chunks));
    let progress = Arc::new(tokio::sync::Mutex::new(0u64));
    
    let results: Vec<Result<()>> = stream::iter(ranges.into_iter().enumerate())
        .map(|(idx, (start, end))| {
            let client = client.clone();
            let url = url.to_owned();
            let path = dest.to_path_buf();
            let semaphore = semaphore.clone();
            let progress = progress.clone();
            let pb = pb.clone();
            
            async move {
                let _permit = semaphore.acquire().await?;
                
                let expected_size = end - start + 1;
                match download_chunk_with_retry(&client, &url, &path, start, end).await {
                    Ok(bytes_written) => {
                        if bytes_written != expected_size {
                            return Err(anyhow::anyhow!(
                                "Chunk {} size mismatch: expected {} bytes, got {}",
                                idx, expected_size, bytes_written
                            ));
                        }
                        
                        let mut prog = progress.lock().await;
                        *prog += bytes_written;
                        pb.set_position(*prog);
                        pb.set_message(format!("Chunk {}/{} complete", idx + 1, config.num_chunks));
                        Ok(())
                    }
                    Err(e) => {
                        pb.set_message(format!("Chunk {} failed", idx + 1));
                        Err(e)
                    }
                }
            }
        })
        .buffer_unordered(config.concurrent_chunks)
        .collect()
        .await;
    
    // Check if all chunks succeeded
    for result in results {
        result?;
    }
    
    let final_size = *progress.lock().await;
    
    // Strict size check - must match exactly
    if final_size != total {
        anyhow::bail!("Downloaded size mismatch: {}/{} bytes", final_size, total);
    }
    
    // Ensure file is properly written and closed
    let file = OpenOptions::new()
        .write(true)
        .open(dest)
        .await?;
    file.sync_all().await?;
    drop(file);
    
    // Verify the file size on disk
    let metadata = fs::metadata(dest).await?;
    let disk_size = metadata.len();
    if disk_size != total {
        anyhow::bail!("File size on disk ({}) doesn't match expected size ({})", disk_size, total);
    }
    
    pb.set_message("Verifying download...");
    
    Ok(())
}

async fn download_chunk_with_retry(
    _client: &Client,
    url: &str,
    path: &Path,
    start: u64,
    end: u64,
) -> Result<u64> {
    let retry_strategy = ExponentialBackoff::from_millis(2000)
        .factor(2)
        .take(3);
    
    Retry::spawn(retry_strategy, || async {
        download_chunk(url, path, start, end).await
    }).await
}

async fn download_chunk(
    url: &str,
    path: &Path,
    start: u64,
    end: u64,
) -> Result<u64> {
    let range_header = format!("bytes={}-{}", start, end);
    let expected_size = end - start + 1;
    
    // Create a new client for this chunk to avoid connection reuse issues
    let chunk_client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CHUNK_TIMEOUT)
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;
    
    let response = chunk_client
        .get(url)
        .header(header::RANGE, &range_header)
        .send()
        .await
        .context("Failed to send request")?;
    
    // Check if server supports range requests
    if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        // Good, server supports ranges
    } else if response.status().is_success() {
        // Server might not support ranges, which is a problem
        return Err(anyhow::anyhow!(
            "Server returned full content instead of range. Status: {}",
            response.status()
        ));
    } else {
        return Err(anyhow::anyhow!(
            "Server returned error status: {}",
            response.status()
        ));
    }
    
    let bytes = response
        .bytes()
        .await
        .context("Failed to download chunk")?;
    
    // Verify we got the expected amount of data
    if bytes.len() as u64 != expected_size {
        return Err(anyhow::anyhow!(
            "Chunk size mismatch: expected {} bytes, got {} bytes for range {}-{}",
            expected_size, bytes.len(), start, end
        ));
    }
    
    // Write to file at correct offset
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .await?;
    
    file.seek(tokio::io::SeekFrom::Start(start)).await?;
    file.write_all(&bytes).await?;
    file.sync_all().await?;
    drop(file);
    
    Ok(bytes.len() as u64)
}

async fn assemble_multipart(first_part: &Path) -> Result<PathBuf> {
    let dir = first_part.parent().unwrap_or(Path::new("."));
    let base_name = first_part
        .file_stem()
        .and_then(|s| s.to_str())
        .context("Invalid filename")?;
    
    // For .zip files, check if there are .z01, .z02 parts
    let mut parts = Vec::new();
    
    // First, add the main zip file
    parts.push(first_part.to_path_buf());
    
    // Then look for part files
    let mut part_num = 1;
    loop {
        let part_name = format!("{}.z{:02}", base_name, part_num);
        let part_path = dir.join(&part_name);
        
        if part_path.exists() {
            parts.push(part_path);
            part_num += 1;
        } else {
            break;
        }
    }
    
    // If no part files found, return the original
    if parts.len() == 1 {
        return Ok(first_part.to_path_buf());
    }
    
    println!("  🔗 Found {} parts for {}", parts.len(), base_name);
    
    // Check all parts exist and have content
    for part in &parts {
        if !part.exists() {
            anyhow::bail!("Missing part file: {}", part.display());
        }
        let metadata = fs::metadata(part).await?;
        if metadata.len() == 0 {
            anyhow::bail!("Empty part file: {}", part.display());
        }
    }
    
    // Assemble into single file
    let output_path = dir.join(format!("{}_complete.zip", base_name));
    let mut output = File::create(&output_path).await?;
    
    for (idx, part) in parts.iter().enumerate() {
        println!("    Assembling part {}/{}", idx + 1, parts.len());
        let mut input = File::open(part).await?;
        tokio::io::copy(&mut input, &mut output).await?;
    }
    
    output.sync_all().await?;
    
    // Clean up part files
    for part in &parts {
        let _ = fs::remove_file(part).await;
    }
    
    Ok(output_path)
}

async fn extract_zip(zip_path: &Path, target_dir: &str) -> Result<()> {
    let zip_path = zip_path.to_owned();
    let target_dir = PathBuf::from(target_dir);
    
    // Create a subdirectory for this dataset
    let dataset_name = zip_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");
    let dataset_dir = target_dir.join(dataset_name);
    
    tokio::task::spawn_blocking(move || -> Result<()> {
        println!("  📂 Opening ZIP file: {}", zip_path.display());
        
        let file = std::fs::File::open(&zip_path)
            .with_context(|| format!("Failed to open ZIP file: {}", zip_path.display()))?;
        
        let file_size = file.metadata()?.len();
        println!("  📊 ZIP file size: {} bytes", file_size);
        
        let mut archive = match ZipArchive::new(file) {
            Ok(archive) => {
                println!("  ✅ Successfully opened ZIP archive");
                archive
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to open ZIP archive: {}. File might be corrupted or incomplete.", e));
            }
        };
        
        // Create dataset directory
        std::fs::create_dir_all(&dataset_dir)?;
        
        let total_files = archive.len();
        println!("  📦 Found {} entries in ZIP file", total_files);
        
        if total_files == 0 {
            println!("  ⚠️  Warning: ZIP file appears to be empty");
            return Ok(());
        }
        
        println!("  📂 Extracting to: {}", dataset_dir.display());
        
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let entry_name = entry.name().to_string();
            let outpath = dataset_dir.join(entry.mangled_name());
            
            if entry.is_dir() {
                std::fs::create_dir_all(&outpath)?;
            } else {
                if let Some(parent) = outpath.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                
                let mut outfile = std::fs::File::create(&outpath)
                    .with_context(|| format!("Failed to create file: {}", outpath.display()))?;
                    
                std::io::copy(&mut entry, &mut outfile)
                    .with_context(|| format!("Failed to extract: {}", entry_name))?;
            }
            
            // Progress indicator every 100 files or at the end
            if (i + 1) % 100 == 0 || i + 1 == total_files {
                println!("    Extracted {}/{} files", i + 1, total_files);
            }
        }
        
        println!("  ✅ Extraction complete: {} files extracted to {}", total_files, dataset_dir.display());
        Ok(())
    })
    .await??;
    
    Ok(())
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{prefix} [{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
        .unwrap()
        .progress_chars("█▓▒░")
}

fn get_extracted_marker_path(zip_path: &Path) -> PathBuf {
    let stem = zip_path.file_stem().unwrap_or_default();
    let marker_name = format!(".{}.extracted", stem.to_string_lossy());
    zip_path.parent().unwrap_or(Path::new(".")).join(marker_name)
}

fn should_check_multipart(path: &Path) -> bool {
    // Only check for multipart if it's a main zip file (not a part file)
    path.extension().and_then(|s| s.to_str()) == Some("zip") &&
    !path.to_string_lossy().contains(".z0")
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    let manifest_path = if args.len() > 1 {
        // Use provided manifest path
        PathBuf::from(&args[1])
    } else {
        // Default to simple manifest
        Path::new("../../test_datasets_simple.json").to_path_buf()
    };
    
    if !manifest_path.exists() {
        eprintln!("❌ Manifest file not found: {}", manifest_path.display());
        eprintln!("Usage: {} [manifest.json]", args[0]);
        eprintln!("Default: ../../test_datasets_simple.json");
        std::process::exit(1);
    }
    
    println!("📋 Using manifest: {}", manifest_path.display());
    process_manifest(&manifest_path).await
}
