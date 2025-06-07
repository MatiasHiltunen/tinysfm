use image::{DynamicImage, GenericImageView, ImageFormat};
use std::path::Path;
use anyhow::{Result, Context};

pub struct ImageLoader {
    supported_formats: Vec<ImageFormat>,
    max_dimension: Option<u32>,
}

impl ImageLoader {
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                ImageFormat::Jpeg,
                ImageFormat::Png,
                ImageFormat::Bmp,
                ImageFormat::Tiff,
                ImageFormat::WebP,
            ],
            max_dimension: None,
        }
    }
    
    pub fn with_max_dimension(mut self, max_dim: u32) -> Self {
        self.max_dimension = Some(max_dim);
        self
    }
    
    pub fn is_supported_format(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            match ext_lower.as_str() {
                "jpg" | "jpeg" => true,
                "png" => true,
                "bmp" => true,
                "tiff" | "tif" => true,
                "webp" => true,
                _ => false,
            }
        } else {
            false
        }
    }
    
    pub fn load(&self, path: &Path) -> Result<ImageData> {
        let img = image::open(path)
            .with_context(|| format!("Failed to load image: {}", path.display()))?;
        
        // Resize if needed
        let img = if let Some(max_dim) = self.max_dimension {
            self.resize_if_needed(img, max_dim)
        } else {
            img
        };
        
        let (width, height) = img.dimensions();
        
        // Convert to grayscale for feature detection
        let gray = img.to_luma8();
        let data: Vec<f32> = gray.pixels()
            .map(|p| p[0] as f32 / 255.0)
            .collect();
        
        Ok(ImageData {
            path: path.to_path_buf(),
            width,
            height,
            data,
            original: img,
        })
    }
    
    fn resize_if_needed(&self, img: DynamicImage, max_dim: u32) -> DynamicImage {
        let (width, height) = img.dimensions();
        let max_current = width.max(height);
        
        if max_current > max_dim {
            let scale = max_dim as f32 / max_current as f32;
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;
            
            img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
        } else {
            img
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImageData {
    pub path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
    pub original: DynamicImage,
}

impl ImageData {
    pub fn extract_exif_focal_length(&self) -> Option<f32> {
        // In a real implementation, we would extract EXIF data
        // For now, return a reasonable default based on image size
        let diagonal = ((self.width * self.width + self.height * self.height) as f32).sqrt();
        Some(diagonal * 0.8) // Approximate focal length
    }
    
    pub fn get_principal_point(&self) -> (f32, f32) {
        (self.width as f32 / 2.0, self.height as f32 / 2.0)
    }
} 