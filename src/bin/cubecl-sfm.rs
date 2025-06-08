use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::{Result, Context};
use indicatif::{ProgressBar, ProgressStyle};
use cubecl_test::{ImageLoader, ImageData, Mast3rPipeline, Mast3rConfig};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;

#[derive(Parser, Debug)]
#[command(name = "cubecl-sfm")]
#[command(author = "CubeCL SfM Team")]
#[command(version = "1.0")]
#[command(about = "GPU-accelerated Structure from Motion pipeline", long_about = None)]
struct Args {
    /// Input folder containing images
    #[arg(short, long)]
    input: PathBuf,

    /// Output folder
    #[arg(short, long, default_value = "nerf_output")]
    output: PathBuf,
    
    /// Output format (nerfstudio)
    #[arg(long, default_value = "nerfstudio")]
    format: String,

    /// Maximum image dimension (images larger than this will be resized)
    #[arg(long, default_value = "4096")]
    max_dimension: u32,

    /// Maximum number of features per image
    #[arg(short = 'f', long, default_value = "5000")]
    max_features: usize,

    /// Number of images to process (0 = all)
    #[arg(short = 'n', long, default_value = "0")]
    num_images: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Dry run - only analyze images without processing
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    
    let args = Args::parse();
    
    println!("🚀 CubeCL Structure from Motion Pipeline");
    println!("========================================\n");
    
    let start_time = Instant::now();
    
    // Step 1: Discover and load images
    let images = load_images(&args)?;
    
    if args.dry_run {
        println!("\n✅ Dry run complete. Found {} images ready for processing.", images.len());
        return Ok(());
    }
    
    // Step 2: Process images
    process_images(&args, &images)?;
    
    let total_time = start_time.elapsed();
    println!("\n✅ Pipeline completed in {:.2}s", total_time.as_secs_f32());
    println!("📊 Performance: {:.1} images/second", 
             images.len() as f32 / total_time.as_secs_f32());
    
    Ok(())
}

fn load_images(args: &Args) -> Result<Vec<ImageData>> {
    println!("📁 Loading images from: {}", args.input.display());
    
    // Check if input directory exists
    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {}", args.input.display());
    }
    
    if !args.input.is_dir() {
        anyhow::bail!("Input path is not a directory: {}", args.input.display());
    }
    
    // Create image loader with max dimension constraint
    let loader = ImageLoader::new()
        .with_max_dimension(args.max_dimension);
    
    // Discover all image files
    let mut image_paths = Vec::new();
    for entry in std::fs::read_dir(&args.input)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && loader.is_supported_format(&path) {
            image_paths.push(path);
        }
    }
    
    if image_paths.is_empty() {
        anyhow::bail!("No supported images found in directory");
    }
    
    // Sort for consistent ordering
    image_paths.sort();
    
    // Limit number of images if requested
    if args.num_images > 0 && image_paths.len() > args.num_images {
        image_paths.truncate(args.num_images);
        println!("📌 Limited to {} images as requested", args.num_images);
    }
    
    println!("🔍 Found {} images to process", image_paths.len());
    
    // Load images with progress bar
    let pb = ProgressBar::new(image_paths.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("#>-"));
    
    let mut images = Vec::new();
    let mut total_pixels: u64 = 0;
    let mut min_width = u32::MAX;
    let mut max_width = 0;
    let mut min_height = u32::MAX;
    let mut max_height = 0;
    
    for path in &image_paths {
        pb.set_message(format!("Loading {}", path.file_name().unwrap().to_string_lossy()));
        
        match loader.load(path) {
            Ok(image_data) => {
                // Update statistics
                total_pixels += (image_data.width * image_data.height) as u64;
                min_width = min_width.min(image_data.width);
                max_width = max_width.max(image_data.width);
                min_height = min_height.min(image_data.height);
                max_height = max_height.max(image_data.height);
                
                if args.verbose {
                    println!("  ✓ {} ({}x{})", 
                        path.file_name().unwrap().to_string_lossy(),
                        image_data.width, 
                        image_data.height);
                }
                
                images.push(image_data);
            }
            Err(e) => {
                eprintln!("  ⚠️  Failed to load {}: {}", 
                    path.file_name().unwrap().to_string_lossy(), e);
            }
        }
        
        pb.inc(1);
    }
    
    pb.finish_with_message("✅ Image loading complete");
    
    if images.is_empty() {
        anyhow::bail!("Failed to load any images");
    }
    
    // Print statistics
    println!("\n📊 Image Statistics:");
    println!("   Successfully loaded: {}/{}", images.len(), image_paths.len());
    println!("   Resolution range: {}×{} to {}×{}", 
             min_width, min_height, max_width, max_height);
    println!("   Total pixels: {:.1}M", total_pixels as f64 / 1_000_000.0);
    let avg_pixels = total_pixels / images.len() as u64;
    let avg_dim = (avg_pixels as f64).sqrt() as u32;
    println!("   Average resolution: {}×{}", avg_dim, avg_dim);
    
    // Check for resolution consistency
    let resolution_variance = (max_width - min_width).max(max_height - min_height);
    if resolution_variance > 1000 {
        println!("   ⚠️  Large resolution variance detected. Consider using --max-dimension flag.");
    }
    
    Ok(images)
}

fn process_images(args: &Args, images: &[ImageData]) -> Result<()> {
    println!("\n🔧 Processing {} images...", images.len());
    
    // Create output directory
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output directory: {}", args.output.display()))?;
    
    // Initialize GPU runtime
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let config = Mast3rConfig {
        max_features: args.max_features,
        ..Default::default()
    };

    let mut pipeline = Mast3rPipeline::<WgpuRuntime>::new(client, config);
    pipeline.run(images)?;

    if args.format.to_lowercase() == "nerfstudio" {
        let path = args.output.join("transforms.json");
        pipeline.export_nerf_transforms(images, &path)?;
        println!("✅ NeRFStudio output written to {}", path.display());
        return Ok(());
    }
    
    if args.format.to_lowercase() != "nerfstudio" {
        println!("⚠️  Only 'nerfstudio' format is supported in this demo");
    }

    Ok(())
}
