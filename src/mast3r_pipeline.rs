use crate::image_loader::ImageData;
use anyhow::Result;
use cubecl::Runtime;
use std::path::Path;
use crate::nerfstudio::{NerfStudioBuilder, Frame};
use serde_json;
use opencv::{prelude::*, core, features2d, calib3d};

/// Configuration for the experimental Mast3r pipeline
#[derive(Debug, Clone)]
pub struct Mast3rConfig {
    pub max_features: usize,
    pub match_ratio: f32,
}

/// Estimated camera intrinsics
#[derive(Debug, Clone)]
pub struct Intrinsics {
    pub fl_x: f64,
    pub fl_y: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Default for Mast3rConfig {
    fn default() -> Self {
        Self {
            max_features: 5000,
            match_ratio: 0.8,
        }
    }
}

/// GPU accelerated pipeline inspired by Mast3rSfM
use cubecl_runtime::client::ComputeClient;

pub struct Mast3rPipeline<R: Runtime> {
    client: ComputeClient<R::Server, R::Channel>,
    config: Mast3rConfig,
    intrinsics: Option<Intrinsics>,
    features: Vec<(core::Vector<core::KeyPoint>, core::Mat)>,
    poses: Vec<core::Mat>,
}

impl<R: Runtime> Mast3rPipeline<R> {
    /// Create a new pipeline using the provided CubeCL runtime
    pub fn new(client: ComputeClient<R::Server, R::Channel>, config: Mast3rConfig) -> Self {
        Self {
            client,
            config,
            intrinsics: None,
            features: Vec::new(),
            poses: Vec::new(),
        }
    }

    /// Estimate shared camera intrinsics from image EXIF data
    fn estimate_intrinsics(&mut self, images: &[ImageData]) -> Result<()> {
        let mut focal_lengths = Vec::new();
        let mut avg_width = 0f64;
        let mut avg_height = 0f64;

        for img in images {
            avg_width += img.width as f64;
            avg_height += img.height as f64;
            if let Some(f) = img.extract_exif_focal_length() {
                focal_lengths.push(f as f64);
            }
        }

        avg_width /= images.len() as f64;
        avg_height /= images.len() as f64;

        let fl = if !focal_lengths.is_empty() {
            focal_lengths.iter().sum::<f64>() / focal_lengths.len() as f64
        } else {
            // Fallback heuristic
            ((avg_width * avg_width + avg_height * avg_height).sqrt()) * 0.8
        };

        self.intrinsics = Some(Intrinsics {
            fl_x: fl,
            fl_y: fl,
            cx: avg_width / 2.0,
            cy: avg_height / 2.0,
        });

        println!(
            "[mast3r] estimated intrinsics: fx={:.2}, fy={:.2}, cx={:.1}, cy={:.1}",
            fl,
            fl,
            avg_width / 2.0,
            avg_height / 2.0
        );
        Ok(())
    }

    /// Main entry point: runs feature extraction, matching and pose estimation
    pub fn run(&mut self, images: &[ImageData]) -> Result<()> {
        self.estimate_intrinsics(images)?;
        self.extract_features(images)?;
        self.estimate_poses()?;
        self.bundle_adjust()?;
        Ok(())
    }

    pub fn intrinsics(&self) -> Option<&Intrinsics> {
        self.intrinsics.as_ref()
    }

    /// Feature extraction placeholder. Real implementation would launch CubeCL
    /// kernels similar to `hierarchical_feature_extraction`.
    fn extract_features(&mut self, images: &[ImageData]) -> Result<()> {
        println!("[mast3r] extracting features with ORB");
        let mut orb = features2d::ORB::create(
            self.config.max_features as i32,
            1.2,
            8,
            31,
            0,
            2,
            features2d::ORB_ScoreType::HARRIS_SCORE,
            31,
            20,
        )?;
        self.features.clear();

        for img in images {
            let mat = opencv::imgcodecs::imread(
                img.path.to_str().unwrap(),
                opencv::imgcodecs::IMREAD_GRAYSCALE,
            )?;
            let mut keypoints = core::Vector::<core::KeyPoint>::new();
            let mut desc = core::Mat::default();
            orb.detect_and_compute(&mat, &core::no_array(), &mut keypoints, &mut desc, false)?;
            self.features.push((keypoints, desc));
        }
        Ok(())
    }

    /// Feature matching placeholder.
    fn match_features(&self) -> Result<Vec<core::Vector<core::DMatch>>> {
        println!("[mast3r] matching features (ratio {})", self.config.match_ratio);
        let mut matches_all = Vec::new();
        if self.features.len() < 2 {
            return Ok(matches_all);
        }
        let bf = features2d::BFMatcher::create(core::NORM_HAMMING, false)?;
        for i in 0..self.features.len() - 1 {
            let desc1 = &self.features[i].1;
            let desc2 = &self.features[i + 1].1;
            let mut knn = core::Vector::<core::Vector<core::DMatch>>::new();
            bf.knn_train_match(desc1, desc2, &mut knn, 2, &core::no_array(), false)?;
            let mut good = core::Vector::<core::DMatch>::new();
            for pair in knn {
                if pair.len() >= 2 {
                    let m = pair.get(0)?;
                    let n = pair.get(1)?;
                    if m.distance < self.config.match_ratio * n.distance {
                        good.push(m);
                    }
                }
            }
            matches_all.push(good);
        }
        Ok(matches_all)
    }

    /// Camera pose estimation placeholder.
    fn estimate_poses(&mut self) -> Result<()> {
        println!("[mast3r] estimating poses using essential matrix");
        let intr = match &self.intrinsics {
            Some(i) => i,
            None => anyhow::bail!("Intrinsics not estimated"),
        };
        let camera_matrix = core::Mat::from_slice_2d(&[
            [intr.fl_x, 0.0, intr.cx],
            [0.0, intr.fl_y, intr.cy],
            [0.0, 0.0, 1.0],
        ])?;

        let matches = self.match_features()?;
        self.poses.clear();
        self.poses.push(core::Mat::eye(3, 4, core::CV_64F)?.to_mat()?); // identity for first frame

        for (i, m) in matches.iter().enumerate() {
            let (kp1, kp2) = (&self.features[i].0, &self.features[i + 1].0);
            let mut pts1 = core::Vector::<core::Point2f>::new();
            let mut pts2 = core::Vector::<core::Point2f>::new();
            for d in m {
                pts1.push(kp1.get(d.query_idx as usize)?.pt());
                pts2.push(kp2.get(d.train_idx as usize)?.pt());
            }
            if pts1.len() < 8 {
                self.poses.push(self.poses[i].clone());
                continue;
            }
            let e = calib3d::find_essential_mat(
                &pts1,
                &pts2,
                &camera_matrix,
                calib3d::RANSAC,
                0.999,
                1.0,
                1000,
                &mut core::no_array(),
            )?;
            let mut r = core::Mat::default();
            let mut t = core::Mat::default();
            calib3d::recover_pose(
                &e,
                &pts1,
                &pts2,
                &mut r,
                &mut t,
                intr.fl_x,
                core::Point2d::new(intr.cx, intr.cy),
                &mut core::no_array(),
            )?;
            // Build 3x4 matrix [R|t]
            let mut rt = core::Mat::zeros(3, 4, core::CV_64F)?.to_mat()?;
            let r_roi = core::Rect::new(0, 0, 3, 3);
            r.copy_to(&mut rt.roi_mut(r_roi)?)?;
            let t_roi = core::Rect::new(3, 0, 1, 3);
            t.copy_to(&mut rt.roi_mut(t_roi)?)?;
            self.poses.push(rt);
        }
        Ok(())
    }

    /// Bundle adjustment placeholder.
    fn bundle_adjust(&self) -> Result<()> {
        println!("[mast3r] running bundle adjustment");
        Ok(())
    }

    /// Export basic NeRFStudio transforms.json
    pub fn export_nerf_transforms<P: AsRef<Path>>(
        &self,
        images: &[ImageData],
        output: P,
    ) -> Result<()> {
        let intr = match &self.intrinsics {
            Some(i) => i,
            None => anyhow::bail!("Intrinsics not estimated"),
        };

        let mut builder = NerfStudioBuilder::new()
            .set_intrinsics(
                intr.fl_x,
                intr.fl_y,
                intr.cx,
                intr.cy,
                images[0].width,
                images[0].height,
            );

        for (i, img) in images.iter().enumerate() {
            let pose = if i < self.poses.len() { &self.poses[i] } else { &self.poses[0] };
            let mut mat_4x4 = [[0.0f64;4];4];
            for r in 0..3 {
                for c in 0..4 {
                    mat_4x4[r][c] = *pose.at_2d::<f64>(r as i32, c as i32)?;
                }
            }
            mat_4x4[3] = [0.0,0.0,0.0,1.0];
            builder = builder.add_frame(Frame {
                file_path: img.path.to_string_lossy().to_string(),
                transform_matrix: mat_4x4,
                fl_x: None,
                fl_y: None,
                depth_file_path: None,
                mask_path: None,
            });
        }

        let transforms = builder.build();
        let json = serde_json::to_string_pretty(&transforms)?;
        std::fs::write(output, json)?;
        Ok(())
    }
}
