use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use anyhow::{Result, Context};
use crate::nerfstudio::{NerfStudioBuilder, Frame, CameraModel, CoordinateConverter};

/// COLMAP camera model
#[derive(Debug, Clone)]
pub struct ColmapCamera {
    pub camera_id: u32,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub params: Vec<f64>,
}

/// COLMAP image with pose
#[derive(Debug, Clone)]
pub struct ColmapImage {
    pub image_id: u32,
    pub qw: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub camera_id: u32,
    pub name: String,
    pub points2d: Vec<(f64, f64, i64)>,
}

/// Parser for COLMAP text files
pub struct ColmapParser;

impl ColmapParser {
    /// Parse cameras.txt file
    pub fn parse_cameras(path: &Path) -> Result<HashMap<u32, ColmapCamera>> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read cameras file: {}", path.display()))?;
        
        let mut cameras = HashMap::new();
        
        for line in content.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 5 {
                continue;
            }
            
            let camera_id = parts[0].parse::<u32>()?;
            let model = parts[1].to_string();
            let width = parts[2].parse::<u32>()?;
            let height = parts[3].parse::<u32>()?;
            
            let params: Vec<f64> = parts[4..]
                .iter()
                .map(|p| p.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()?;
            
            cameras.insert(camera_id, ColmapCamera {
                camera_id,
                model,
                width,
                height,
                params,
            });
        }
        
        Ok(cameras)
    }
    
    /// Parse images.txt file
    pub fn parse_images(path: &Path) -> Result<Vec<ColmapImage>> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read images file: {}", path.display()))?;
        
        let mut images = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        
        while i < lines.len() {
            let line = lines[i];
            
            // Skip comments and empty lines
            if line.starts_with('#') || line.trim().is_empty() {
                i += 1;
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            // Check if this is an image line (should have at least 10 parts)
            if parts.len() >= 10 {
                // Try to parse as image data
                if let Ok(image_id) = parts[0].parse::<u32>() {
                    let qw = parts[1].parse::<f64>()?;
                    let qx = parts[2].parse::<f64>()?;
                    let qy = parts[3].parse::<f64>()?;
                    let qz = parts[4].parse::<f64>()?;
                    let tx = parts[5].parse::<f64>()?;
                    let ty = parts[6].parse::<f64>()?;
                    let tz = parts[7].parse::<f64>()?;
                    let camera_id = parts[8].parse::<u32>()?;
                    let name = parts[9].to_string();
                    
                    images.push(ColmapImage {
                        image_id,
                        qw, qx, qy, qz,
                        tx, ty, tz,
                        camera_id,
                        name,
                        points2d: Vec::new(),
                    });
                    
                    // Skip the next line (points2D)
                    i += 2;
                    continue;
                }
            }
            
            i += 1;
        }
        
        Ok(images)
    }
}

/// Converter from COLMAP to NeRFStudio format
pub struct ColmapToNerfConverter {
    colmap_dir: PathBuf,
    image_dir: PathBuf,
    output_dir: PathBuf,
}

impl ColmapToNerfConverter {
    pub fn new(colmap_dir: PathBuf, image_dir: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            colmap_dir,
            image_dir,
            output_dir,
        }
    }
    
    /// Convert COLMAP output to NeRFStudio format
    pub fn convert(&self) -> Result<()> {
        println!("🔄 Converting COLMAP to NeRFStudio format...");
        
        // Parse COLMAP files
        let cameras = ColmapParser::parse_cameras(&self.colmap_dir.join("cameras.txt"))?;
        let images = ColmapParser::parse_images(&self.colmap_dir.join("images.txt"))?;
        
        if cameras.is_empty() {
            anyhow::bail!("No cameras found in COLMAP output");
        }
        
        if images.is_empty() {
            anyhow::bail!("No images found in COLMAP output");
        }
        
        // Assume single camera for now (common case)
        let camera = cameras.values().next().unwrap();
        
        // Extract intrinsics based on camera model
        let (fl_x, fl_y, cx, cy) = match camera.model.as_str() {
            "PINHOLE" => {
                if camera.params.len() < 4 {
                    anyhow::bail!("Invalid PINHOLE camera parameters");
                }
                (camera.params[0], camera.params[1], camera.params[2], camera.params[3])
            }
            "SIMPLE_PINHOLE" => {
                if camera.params.len() < 3 {
                    anyhow::bail!("Invalid SIMPLE_PINHOLE camera parameters");
                }
                let f = camera.params[0];
                (f, f, camera.params[1], camera.params[2])
            }
            _ => anyhow::bail!("Unsupported camera model: {}", camera.model),
        };
        
        // Create NeRFStudio builder
        let mut builder = NerfStudioBuilder::new()
            .set_camera_model(CameraModel::Opencv)
            .set_intrinsics(fl_x, fl_y, cx, cy, camera.width, camera.height);
        
        // Convert each image
        for (idx, image) in images.iter().enumerate() {
            // Convert quaternion to rotation matrix
            let rotation = CoordinateConverter::quaternion_to_matrix(
                image.qw, image.qx, image.qy, image.qz
            );
            
            // Create translation vector
            let translation = nalgebra::Vector3::new(image.tx, image.ty, image.tz);
            
            // Convert to NeRF coordinate system
            let transform = CoordinateConverter::colmap_to_nerf_camera(&rotation, &translation);
            
            // Convert to nested array format
            let mut transform_array = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    transform_array[i][j] = transform[(i, j)];
                }
            }
            
            // Create relative path for image
            let image_path = format!("images/{}", image.name);
            
            builder = builder.add_frame(Frame {
                file_path: image_path,
                transform_matrix: transform_array,
                fl_x: None, // Using global intrinsics
                fl_y: None,
                depth_file_path: None,
                mask_path: None,
            });
            
            if (idx + 1) % 10 == 0 {
                println!("  Converted {}/{} images", idx + 1, images.len());
            }
        }
        
        let transforms = builder.build();
        
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;
        
        // Write transforms.json
        let transforms_path = self.output_dir.join("transforms.json");
        let json = serde_json::to_string_pretty(&transforms)?;
        fs::write(&transforms_path, json)?;
        
        println!("✅ Created transforms.json with {} frames", transforms.frames.len());
        
        // Create images directory in output
        let output_images_dir = self.output_dir.join("images");
        fs::create_dir_all(&output_images_dir)?;
        
        // Copy or link images
        self.setup_images(&images, &output_images_dir)?;
        
        Ok(())
    }
    
    /// Copy or symlink images to output directory
    fn setup_images(&self, images: &[ColmapImage], output_images_dir: &Path) -> Result<()> {
        println!("📷 Setting up images...");
        
        for image in images {
            let src = self.image_dir.join(&image.name);
            let dst = output_images_dir.join(&image.name);
            
            if src.exists() {
                // Try to create symlink first (more efficient)
                #[cfg(unix)]
                {
                    if std::os::unix::fs::symlink(&src, &dst).is_ok() {
                        continue;
                    }
                }
                
                #[cfg(windows)]
                {
                    if std::os::windows::fs::symlink_file(&src, &dst).is_ok() {
                        continue;
                    }
                }
                
                // Fall back to copying
                fs::copy(&src, &dst)
                    .with_context(|| format!("Failed to copy image: {}", src.display()))?;
            } else {
                eprintln!("⚠️  Image not found: {}", src.display());
            }
        }
        
        println!("✅ Images ready in output directory");
        Ok(())
    }
}

/// CLI integration for COLMAP to NeRF conversion
pub fn convert_colmap_to_nerf(
    colmap_dir: &Path,
    image_dir: &Path,
    output_dir: &Path,
) -> Result<()> {
    let converter = ColmapToNerfConverter::new(
        colmap_dir.to_path_buf(),
        image_dir.to_path_buf(),
        output_dir.to_path_buf(),
    );
    
    converter.convert()
} 