#[cfg(test)]
mod tests {
    use std::path::Path;

    #[test]
    fn test_coordinate_conversion() {
        println!("🧪 Testing coordinate system conversions...");
        
        use cubecl_test::nerfstudio::CoordinateConverter;
        use nalgebra::{Matrix3, Vector3};
        
        // Test identity transformation
        let identity = Matrix3::identity();
        let zero_translation = Vector3::zeros();
        let result = CoordinateConverter::colmap_to_nerf_camera(&identity, &zero_translation);
        
        // COLMAP to NeRF conversion should flip Y and Z axes
        // COLMAP: +Y down, +Z forward -> NeRF: +Y up, +Z back
        assert!((result[(0, 0)] - 1.0).abs() < 1e-6, "X axis should remain unchanged");
        assert!((result[(1, 1)] - (-1.0)).abs() < 1e-6, "Y axis should be flipped"); 
        assert!((result[(2, 2)] - (-1.0)).abs() < 1e-6, "Z axis should be flipped");
        
        // Test with translation vector
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let result_with_translation = CoordinateConverter::colmap_to_nerf_camera(&identity, &translation);
        
        // Translation should also be transformed
        assert!((result_with_translation[(0, 3)] - 1.0).abs() < 1e-6, "X translation unchanged");
        assert!((result_with_translation[(1, 3)] - (-2.0)).abs() < 1e-6, "Y translation flipped");
        assert!((result_with_translation[(2, 3)] - (-3.0)).abs() < 1e-6, "Z translation flipped");
        
        // Test quaternion conversion
        let (qw, qx, qy, qz) = (1.0, 0.0, 0.0, 0.0); // Identity quaternion
        let rotation_matrix = CoordinateConverter::quaternion_to_matrix(qw, qx, qy, qz);
        
        assert!((rotation_matrix[(0, 0)] - 1.0).abs() < 1e-6, "Identity quaternion should give identity matrix");
        assert!((rotation_matrix[(1, 1)] - 1.0).abs() < 1e-6, "Identity quaternion should give identity matrix");
        assert!((rotation_matrix[(2, 2)] - 1.0).abs() < 1e-6, "Identity quaternion should give identity matrix");
        
        println!("✅ Coordinate conversion test passed!");
    }

    #[test]
    fn test_image_loader_creation() {
        println!("🧪 Testing image loader creation...");
        
        use cubecl_test::image_loader::ImageLoader;
        
        let _loader = ImageLoader::new();
        
        // Test that we can create an image loader
        println!("  ✓ ImageLoader created successfully");
        
        // Test with existing test images directory
        let test_images_dir = Path::new("test_images");
        if test_images_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(test_images_dir) {
                let image_count = entries
                    .filter_map(|entry| {
                        let entry = entry.ok()?;
                        let path = entry.path();
                        if path.is_file() && path.extension()?.to_str()? == "jpg" {
                            Some(path)
                        } else {
                            None
                        }
                    })
                    .count();
                
                if image_count > 0 {
                    println!("  ✓ Found {} test images for potential loading", image_count);
                } else {
                    println!("  ⚠️ No test images found");
                }
            }
        } else {
            println!("  ⚠️ test_images directory not found");
        }
        
        println!("✅ Image loader test passed!");
    }

    #[test]
    fn test_production_demo_module() {
        println!("🧪 Testing production demo module...");
        
        // Test that we can call the production demo
        cubecl_test::production_demo::run_production_demo();
        
        // Verify output files were created
        assert!(Path::new("colmap_output/cameras.txt").exists(), "cameras.txt should exist");
        assert!(Path::new("colmap_output/images.txt").exists(), "images.txt should exist");
        assert!(Path::new("colmap_output/points3D.txt").exists(), "points3D.txt should exist");
        
        println!("✅ Production demo test passed!");
    }

    #[test]
    fn test_colmap_parser() {
        println!("🧪 Testing COLMAP parser...");
        
        use cubecl_test::colmap_to_nerf::ColmapParser;
        
        // Test parsing the generated cameras.txt
        if Path::new("colmap_output/cameras.txt").exists() {
            match ColmapParser::parse_cameras(Path::new("colmap_output/cameras.txt")) {
                Ok(cameras) => {
                    assert!(!cameras.is_empty(), "Should parse at least one camera");
                    println!("  ✓ Parsed {} cameras", cameras.len());
                    
                    // Check first camera
                    if let Some(camera) = cameras.values().next() {
                        assert_eq!(camera.model, "PINHOLE", "Should be PINHOLE model");
                        assert_eq!(camera.width, 640, "Width should be 640");
                        assert_eq!(camera.height, 480, "Height should be 480");
                        println!("  ✓ Camera parameters valid");
                    }
                },
                Err(e) => {
                    panic!("Failed to parse cameras.txt: {}", e);
                }
            }
        } else {
            println!("  ⚠️ cameras.txt not found, skipping parser test");
        }
        
        // Test parsing images.txt
        if Path::new("colmap_output/images.txt").exists() {
            match ColmapParser::parse_images(Path::new("colmap_output/images.txt")) {
                Ok(images) => {
                    assert!(!images.is_empty(), "Should parse at least one image");
                    println!("  ✓ Parsed {} images", images.len());
                },
                Err(e) => {
                    panic!("Failed to parse images.txt: {}", e);
                }
            }
        }
        
        println!("✅ COLMAP parser test passed!");
    }

    #[test]
    fn test_nerfstudio_builder() {
        println!("🧪 Testing NeRFStudio builder...");
        
        use cubecl_test::nerfstudio::{NerfStudioBuilder, Frame};
        
        let mut builder = NerfStudioBuilder::new()
            .set_intrinsics(512.0, 384.0, 320.0, 240.0, 640, 480);
        
        // Add a test frame
        let transform = [[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]];
        
        builder = builder.add_frame(Frame {
            file_path: "test_image.jpg".to_string(),
            transform_matrix: transform,
            fl_x: None,
            fl_y: None,
            depth_file_path: None,
            mask_path: None,
        });
        
        let transforms = builder.build();
        
        assert_eq!(transforms.frames.len(), 1);
        assert_eq!(transforms.fl_x, Some(512.0));
        assert_eq!(transforms.fl_y, Some(384.0));
        
        println!("  ✓ NeRFStudio transforms built successfully");
        println!("✅ NeRFStudio builder test passed!");
    }

    #[test]
    fn test_mast3r_pipeline_basic() {
        use cubecl_test::{ImageLoader, Mast3rPipeline, Mast3rConfig};
        use cubecl::wgpu::{WgpuRuntime, WgpuDevice};
        use cubecl::Runtime;

        let loader = ImageLoader::new().with_max_dimension(256);
        let mut paths: Vec<_> = std::fs::read_dir("test_images")
            .expect("test_images dir")
            .take(2)
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        let images: Vec<_> = paths.iter().map(|p| loader.load(p).unwrap()).collect();

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let mut pipeline = Mast3rPipeline::<WgpuRuntime>::new(client, Mast3rConfig::default());
        pipeline.run(&images).unwrap();

        assert!(pipeline.intrinsics().is_some());

        let tmp = tempfile::NamedTempFile::new().unwrap();
        pipeline.export_nerf_transforms(&images, tmp.path()).unwrap();
    }
} 