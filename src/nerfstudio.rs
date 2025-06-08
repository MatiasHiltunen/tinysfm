use serde::{Deserialize, Serialize};
use nalgebra::{Matrix3, Matrix4, Quaternion, UnitQuaternion, Vector3};

/// Camera model types supported by NeRFStudio
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CameraModel {
    Opencv,
    OpencvFisheye,
}

/// NeRFStudio transforms.json format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerfStudioTransforms {
    /// Camera model type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_model: Option<CameraModel>,
    
    /// Focal length X (shared intrinsics)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_x: Option<f64>,
    
    /// Focal length Y (shared intrinsics)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_y: Option<f64>,
    
    /// Principal point X
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cx: Option<f64>,
    
    /// Principal point Y
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cy: Option<f64>,
    
    /// Image width
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w: Option<u32>,
    
    /// Image height
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h: Option<u32>,
    
    /// Radial distortion parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k3: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k4: Option<f64>,
    
    /// Tangential distortion parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2: Option<f64>,
    
    /// Per-frame data
    pub frames: Vec<Frame>,
}

/// Per-frame data in transforms.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Path to image file
    pub file_path: String,
    
    /// 4x4 transformation matrix [R|t]
    pub transform_matrix: [[f64; 4]; 4],
    
    /// Per-frame focal length X (overrides global)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_x: Option<f64>,
    
    /// Per-frame focal length Y (overrides global)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fl_y: Option<f64>,
    
    /// Optional depth file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_file_path: Option<String>,
    
    /// Optional mask file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path: Option<String>,
}

/// Coordinate system converter between COLMAP and NeRF conventions
pub struct CoordinateConverter;

impl CoordinateConverter {
    /// Convert COLMAP/OpenCV camera to NeRF/OpenGL camera
    /// COLMAP: +Y down, +Z forward
    /// NeRF: +Y up, +Z back
    pub fn colmap_to_nerf_camera(rotation: &Matrix3<f64>, translation: &Vector3<f64>) -> Matrix4<f64> {
        // Create the coordinate system transformation
        // Flip Y and Z axes
        let flip = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
        );
        
        // Apply transformation: R_nerf = flip * R_colmap
        let r_nerf = flip * rotation;
        let t_nerf = flip * translation;
        
        // Build 4x4 transformation matrix
        let mut transform = Matrix4::identity();
        transform.fixed_slice_mut::<3, 3>(0, 0).copy_from(&r_nerf);
        transform.fixed_slice_mut::<3, 1>(0, 3).copy_from(&t_nerf);
        
        transform
    }
    
    /// Convert quaternion rotation to rotation matrix
    pub fn quaternion_to_matrix(qw: f64, qx: f64, qy: f64, qz: f64) -> Matrix3<f64> {
        let q = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
        q.to_rotation_matrix().into_inner()
    }
    
    /// Convert COLMAP camera parameters to NeRF intrinsics
    pub fn colmap_to_nerf_intrinsics(
        focal_length: f64,
        principal_point_x: f64,
        principal_point_y: f64,
        _width: u32,
        _height: u32,
    ) -> (f64, f64, f64, f64) {
        // COLMAP uses normalized coordinates, NeRF uses pixel coordinates
        let fl_x = focal_length;
        let fl_y = focal_length;
        let cx = principal_point_x;
        let cy = principal_point_y;
        
        (fl_x, fl_y, cx, cy)
    }
}

/// Builder for creating NeRFStudio transforms
pub struct NerfStudioBuilder {
    transforms: NerfStudioTransforms,
}

impl NerfStudioBuilder {
    pub fn new() -> Self {
        Self {
            transforms: NerfStudioTransforms {
                camera_model: Some(CameraModel::Opencv),
                fl_x: None,
                fl_y: None,
                cx: None,
                cy: None,
                w: None,
                h: None,
                k1: Some(0.0),
                k2: Some(0.0),
                k3: None,
                k4: None,
                p1: Some(0.0),
                p2: Some(0.0),
                frames: Vec::new(),
            },
        }
    }
    
    pub fn set_camera_model(mut self, model: CameraModel) -> Self {
        self.transforms.camera_model = Some(model);
        self
    }
    
    pub fn set_intrinsics(mut self, fl_x: f64, fl_y: f64, cx: f64, cy: f64, w: u32, h: u32) -> Self {
        self.transforms.fl_x = Some(fl_x);
        self.transforms.fl_y = Some(fl_y);
        self.transforms.cx = Some(cx);
        self.transforms.cy = Some(cy);
        self.transforms.w = Some(w);
        self.transforms.h = Some(h);
        self
    }
    
    pub fn add_frame(mut self, frame: Frame) -> Self {
        self.transforms.frames.push(frame);
        self
    }
    
    pub fn build(self) -> NerfStudioTransforms {
        self.transforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coordinate_conversion() {
        // Test identity rotation
        let rotation = Matrix3::identity();
        let translation = Vector3::new(1.0, 2.0, 3.0);
        
        let nerf_transform = CoordinateConverter::colmap_to_nerf_camera(&rotation, &translation);
        
        // Check that Y and Z are flipped
        assert_eq!(nerf_transform[(0, 3)], 1.0); // X unchanged
        assert_eq!(nerf_transform[(1, 3)], -2.0); // Y flipped
        assert_eq!(nerf_transform[(2, 3)], -3.0); // Z flipped
    }
    
    #[test]
    fn test_serialization() {
        let transforms = NerfStudioBuilder::new()
            .set_intrinsics(1000.0, 1000.0, 500.0, 500.0, 1000, 1000)
            .add_frame(Frame {
                file_path: "images/frame_00001.jpg".to_string(),
                transform_matrix: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                fl_x: None,
                fl_y: None,
                depth_file_path: None,
                mask_path: None,
            })
            .build();
        
        let json = serde_json::to_string_pretty(&transforms).unwrap();
        assert!(json.contains("\"camera_model\": \"OPENCV\""));
        assert!(json.contains("\"fl_x\": 1000.0"));
    }
} 