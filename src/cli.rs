use clap::{Parser, ValueEnum};
use image::{DynamicImage, GenericImageView};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Instant;
use anyhow::{Result, Context};

use crate::sfm_pipeline::{SfMPipeline, PipelineConfig};
use crate::feature_detector::{FeatureDetector, DetectorType};
use crate::image_loader::ImageLoader;

#[derive(Parser, Debug)]
#[command(name = "cubecl-sfm")]
#[command(author = "CubeCL SfM Team")]
#[command(version = "1.0")]
#[command(about = "GPU-accelerated Structure from Motion pipeline for Gaussian Splatting", long_about = None)]
struct Args {
    /// Input folder containing images
    #[arg(short, long)]
    input: PathBuf,

    /// Output folder for COLMAP files
    #[arg(short, long, default_value = "colmap_output")]
    output: PathBuf,

    /// Feature detector type
    #[arg(short = 'd', long, value_enum, default_value = "sift")]
    detector: DetectorType,

    /// Maximum number of features per image
    #[arg(short = 'f', long, default_value = "5000")]
    max_features: usize,

    /// Feature matching ratio threshold
    #[arg(short = 'r', long, default_value = "0.8")]
    match_ratio: f32,

    /// RANSAC threshold for essential matrix
    #[arg(long, default_value = "1.0")]
    ransac_threshold: f32,

    /// Bundle adjustment iterations
    #[arg(long, default_value = "100")]
    ba_iterations: usize,

    /// GPU device index
    #[arg(short = 'g', long, default_value = "0")]
    gpu_device: usize,

    /// Quality preset
    #[arg(short = 'q', long, value_enum, default_value = "balanced")]
    quality: QualityPreset,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Process images in batches (for memory efficiency)
    #[arg(short = 'b', long, default_value = "50")]
    batch_size: usize,

    /// Enable incremental reconstruction
    #[arg(long)]
    incremental: bool,

    /// Camera model
    #[arg(long, value_enum, default_value = "pinhole")]
    camera_model: CameraModel,

    /// Undistort images before processing
    #[arg(long)]
    undistort: bool,

    /// Export sparse point cloud in PLY format
    #[arg(long)]
    export_ply: bool,

    /// Export camera trajectory
    #[arg(long)]
    export_trajectory: bool,

    /// Number of threads for CPU operations
    #[arg(short = 't', long, default_value = "0")]
    threads: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum QualityPreset {
    /// Fast processing, lower quality
    Fast,
    /// Balanced speed and quality
    Balanced,
    /// High quality, slower processing
    High,
    /// Maximum quality for production
    Ultra,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum CameraModel {
    /// Simple pinhole camera model
    Pinhole,
    /// Pinhole with radial distortion
    PinholeRadial,
    /// Full OpenCV camera model
    OpenCV,
}

pub struct CLIApp {
    args: Args,
    pipeline: SfMPipeline,
    image_loader: ImageLoader,
}

impl CLIApp {
    pub fn new() -> Result<Self> {
        let args = Args::parse();
        
        // Set thread count
        if args.threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.threads)
                .build_global()
                .context("Failed to set thread pool")?;
        }
        
        // Create pipeline configuration based on quality preset
        let config = match args.quality {
            QualityPreset::Fast => PipelineConfig {
                max_features: 2000,
                feature_quality: 0.01,
                match_ratio: 0.9,
                ransac_iterations: 500,
                ba_iterations: 20,
                ..Default::default()
            },
            QualityPreset::Balanced => PipelineConfig {
                max_features: args.max_features,
                feature_quality: 0.03,
                match_ratio: args.match_ratio,
                ransac_iterations: 1000,
                ba_iterations: args.ba_iterations,
                ..Default::default()
            },
            QualityPreset::High => PipelineConfig {
                max_features: 8000,
                feature_quality: 0.05,
                match_ratio: 0.7,
                ransac_iterations: 2000,
                ba_iterations: 200,
                ..Default::default()
            },
            QualityPreset::Ultra => PipelineConfig {
                max_features: 15000,
                feature_quality: 0.1,
                match_ratio: 0.6,
                ransac_iterations: 5000,
                ba_iterations: 500,
                ..Default::default()
            },
        };
        
        let pipeline = SfMPipeline::new(config, args.gpu_device)?;
        let image_loader = ImageLoader::new();
        
        Ok(Self {
            args,
            pipeline,
            image_loader,
        })
    }
    
    pub fn run(&mut self) -> Result<()> {
        println!("🚀 CubeCL Structure from Motion Pipeline");
        println!("========================================\n");
        
        let start_time = Instant::now();
        
        // Step 1: Load and validate images
        let images = self.load_images()?;
        println!("✅ Loaded {} images", images.len());
        
        // Step 2: Extract features
        self.extract_features(&images)?;
        
        // Step 3: Match features
        self.match_features()?;
        
        // Step 4: Reconstruct
        if self.args.incremental {
            self.incremental_reconstruction()?;
        } else {
            self.global_reconstruction()?;
        }
        
        // Step 5: Bundle adjustment
        self.bundle_adjustment()?;
        
        // Step 6: Export results
        self.export_results()?;
        
        let total_time = start_time.elapsed();
        println!("\n✅ Pipeline completed in {:.2}s", total_time.as_secs_f32());
        println!("📊 Performance: {:.1} images/second", 
                 images.len() as f32 / total_time.as_secs_f32());
        
        Ok(())
    }
    
    fn load_images(&self) -> Result<Vec<ImageData>> {
        println!("📁 Loading images from: {}", self.args.input.display());
        
        let mut images = Vec::new();
        let entries = fs::read_dir(&self.args.input)
            .context("Failed to read input directory")?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if self.image_loader.is_supported_format(&path) {
                if self.args.verbose {
                    println!("  Loading: {}", path.file_name().unwrap().to_string_lossy());
                }
                
                let image_data = self.image_loader.load(&path)?;
                images.push(image_data);
            }
        }
        
        if images.is_empty() {
            anyhow::bail!("No supported images found in input directory");
        }
        
        // Sort images by name for consistent ordering
        images.sort_by(|a, b| a.path.cmp(&b.path));
        
        // Print image statistics
        let (min_width, max_width) = images.iter()
            .map(|img| img.width)
            .fold((u32::MAX, 0), |(min, max), w| (min.min(w), max.max(w)));
        
        let (min_height, max_height) = images.iter()
            .map(|img| img.height)
            .fold((u32::MAX, 0), |(min, max), h| (min.min(h), max.max(h)));
        
        println!("📊 Image statistics:");
        println!("   Resolution range: {}x{} to {}x{}", 
                 min_width, min_height, max_width, max_height);
        println!("   Total pixels: {:.1}M", 
                 images.iter().map(|img| img.width * img.height).sum::<u32>() as f32 / 1_000_000.0);
        
        Ok(images)
    }
    
    fn extract_features(&mut self, images: &[ImageData]) -> Result<()> {
        println!("\n🔍 Extracting features using {} detector...", 
                 format!("{:?}", self.args.detector).to_lowercase());
        
        let progress = indicatif::ProgressBar::new(images.len() as u64);
        progress.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-")
        );
        
        // Process in batches for memory efficiency
        for batch in images.chunks(self.args.batch_size) {
            self.pipeline.extract_features_batch(batch, self.args.detector)?;
            progress.inc(batch.len() as u64);
        }
        
        progress.finish_with_message("✅ Feature extraction complete");
        
        let total_features = self.pipeline.get_total_features();
        println!("   Extracted {} total features", total_features);
        println!("   Average features per image: {}", total_features / images.len());
        
        Ok(())
    }
    
    fn match_features(&mut self) -> Result<()> {
        println!("\n🔗 Matching features across images...");
        
        let num_pairs = self.pipeline.match_all_features()?;
        
        println!("   Found {} valid image pairs", num_pairs);
        println!("   Average matches per pair: {}", 
                 self.pipeline.get_average_matches_per_pair());
        
        Ok(())
    }
    
    fn incremental_reconstruction(&mut self) -> Result<()> {
        println!("\n🎯 Running incremental reconstruction...");
        
        let result = self.pipeline.reconstruct_incremental()?;
        
        println!("   Reconstructed {} cameras", result.num_cameras);
        println!("   Triangulated {} 3D points", result.num_points);
        println!("   Average reprojection error: {:.3} pixels", result.avg_reprojection_error);
        
        Ok(())
    }
    
    fn global_reconstruction(&mut self) -> Result<()> {
        println!("\n🌐 Running global reconstruction...");
        
        let result = self.pipeline.reconstruct_global()?;
        
        println!("   Reconstructed {} cameras", result.num_cameras);
        println!("   Triangulated {} 3D points", result.num_points);
        println!("   Average reprojection error: {:.3} pixels", result.avg_reprojection_error);
        
        Ok(())
    }
    
    fn bundle_adjustment(&mut self) -> Result<()> {
        println!("\n⚡ Running bundle adjustment...");
        
        let initial_error = self.pipeline.get_reprojection_error();
        let final_error = self.pipeline.bundle_adjust(self.args.ba_iterations)?;
        
        println!("   Initial error: {:.3} pixels", initial_error);
        println!("   Final error: {:.3} pixels", final_error);
        println!("   Improvement: {:.1}%", 
                 (1.0 - final_error / initial_error) * 100.0);
        
        Ok(())
    }
    
    fn export_results(&self) -> Result<()> {
        println!("\n📁 Exporting results to: {}", self.args.output.display());
        
        // Create output directory
        fs::create_dir_all(&self.args.output)?;
        
        // Export COLMAP files
        self.pipeline.export_colmap(&self.args.output)?;
        println!("   ✅ Exported COLMAP files (cameras.txt, images.txt, points3D.txt)");
        
        // Export PLY if requested
        if self.args.export_ply {
            let ply_path = self.args.output.join("sparse_cloud.ply");
            self.pipeline.export_ply(&ply_path)?;
            println!("   ✅ Exported sparse point cloud: sparse_cloud.ply");
        }
        
        // Export camera trajectory if requested
        if self.args.export_trajectory {
            let traj_path = self.args.output.join("camera_trajectory.txt");
            self.pipeline.export_trajectory(&traj_path)?;
            println!("   ✅ Exported camera trajectory: camera_trajectory.txt");
        }
        
        // Generate reconstruction statistics
        let stats_path = self.args.output.join("reconstruction_stats.json");
        self.pipeline.export_statistics(&stats_path)?;
        println!("   ✅ Exported statistics: reconstruction_stats.json");
        
        Ok(())
    }
}

#[derive(Debug)]
pub struct ImageData {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>, // Normalized grayscale data for GPU
    pub original: DynamicImage,
}

// Main entry point
pub fn main() -> Result<()> {
    env_logger::init();
    
    let mut app = CLIApp::new()?;
    app.run()?;
    
    Ok(())
} 