use std::fs;
use std::path::Path;
use std::time::Instant;
use cubecl_test::*;

fn main() {
    println!("🧪 === CubeCL SfM Pipeline Test Suite ===\n");
    
    let overall_start = Instant::now();
    let mut tests_passed = 0;
    let mut tests_failed = 0;
    
    // Test basic pipeline
    println!("📋 Test: Basic Pipeline");
    if test_basic_pipeline() {
        println!("✅ Basic pipeline test passed!\n");
        tests_passed += 1;
    } else {
        println!("❌ Basic pipeline test failed!\n");
        tests_failed += 1;
    }
    
    // Test real images
    println!("📋 Test: Real Images");
    if test_real_images() {
        println!("✅ Real image test passed!\n");
        tests_passed += 1;
    } else {
        println!("❌ Real image test failed!\n");
        tests_failed += 1;
    }
    
    // Test COLMAP format
    println!("📋 Test: COLMAP Format");
    if test_colmap_format() {
        println!("✅ COLMAP format test passed!\n");
        tests_passed += 1;
    } else {
        println!("❌ COLMAP format test failed!\n");
        tests_failed += 1;
    }
    
    let total_time = overall_start.elapsed();
    
    println!("🏁 Test Results:");
    println!("  ✅ Passed: {}", tests_passed);
    println!("  ❌ Failed: {}", tests_failed);
    println!("  ⏱️  Total time: {:.2}s", total_time.as_secs_f64());
    
    if tests_failed == 0 {
        println!("\n🎉 All tests passed!");
    } else {
        println!("\n💥 Some tests failed.");
    }
}

fn test_basic_pipeline() -> bool {
    println!("  🔄 Running production pipeline...");
    
    production_demo::run_production_demo();
    
    Path::new("colmap_output/cameras.txt").exists() &&
    Path::new("colmap_output/images.txt").exists() &&
    Path::new("colmap_output/points3D.txt").exists()
}

fn test_real_images() -> bool {
    let test_images_dir = Path::new("test_images");
    
    if !test_images_dir.exists() {
        println!("  ⚠️ test_images directory not found");
        return false;
    }
    
    if let Ok(entries) = fs::read_dir(test_images_dir) {
        let count = entries.filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() && path.extension()?.to_str()? == "jpg" {
                Some(path)
            } else {
                None
            }
        }).count();
        
        println!("  📷 Found {} test images", count);
        count > 0
    } else {
        false
    }
}

fn test_colmap_format() -> bool {
    println!("  🔍 Validating COLMAP output...");
    
    // Check cameras.txt
    if let Ok(content) = fs::read_to_string("colmap_output/cameras.txt") {
        if !content.contains("PINHOLE") {
            println!("    ❌ cameras.txt missing PINHOLE model");
            return false;
        }
        println!("    ✓ cameras.txt valid");
    } else {
        return false;
    }
    
    // Check images.txt 
    if let Ok(content) = fs::read_to_string("colmap_output/images.txt") {
        let count = content.lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .filter(|line| line.split_whitespace().count() >= 10)
            .count();
        
        if count < 10 {
            println!("    ❌ Expected at least 10 images, found {}", count);
            return false;
        }
        println!("    ✓ images.txt valid ({} images)", count);
    } else {
        return false;
    }
    
    true
} 