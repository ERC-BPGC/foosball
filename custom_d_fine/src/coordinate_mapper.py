import cv2
import numpy as np
import json
from pathlib import Path

class CoordinateMapper:
    def __init__(self, cam_id, side):
        self.side = side
        self.cam_id = cam_id
        
        # Paths to your config files
        # These are the files you created in Phase 1 and with the GUI
        self.lens_path = Path(f"config/lens_intrinsics_{side}.json")
        self.field_path = Path(f"config/calibration_{cam_id}.json")
        
        self._load_configs()

    def _load_configs(self):
        # 1. Load Lens Intrinsics (K and D)
        # K = Camera Matrix (Focal length, optical center)
        # D = Distortion Coefficients (How much the lens bends light)
        if not self.lens_path.exists():
            print(f"Warning: Lens config not found for {self.side}. Using identity.")
            self.K = np.eye(3)
            self.D = np.zeros(5)
        else:
            with open(self.lens_path, "r") as f:
                lens_data = json.load(f)
                self.K = np.array(lens_data["camera_matrix"], dtype=np.float64)
                self.D = np.array(lens_data["dist_coeffs"], dtype=np.float64)

        # 2. Load Field Homography (H)
        # H = Homography Matrix (Maps pixels to millimeters)
        if not self.field_path.exists():
            raise FileNotFoundError(f"CRITICAL: Missing field calibration for {self.side} (ID {self.cam_id}). Run the GUI first!")
            
        with open(self.field_path, "r") as f:
            field_data = json.load(f)
            self.H = np.array(field_data["homography_matrix"], dtype=np.float64)
            
        print(f"[{self.side.upper()}] Mapper Initialized.")

    def pixel_to_world(self, u, v):
        """
        Converts raw pixel (u, v) -> World (x_mm, y_mm)
        """
        # Step 1: Undistort Point
        # We assume the point is a 2D pixel coordinate.
        # This function corrects the "fisheye" effect.
        src_pt = np.array([[[u, v]]], dtype=np.float64)
        
        # P=self.K ensures we get back pixel coordinates, not normalized ones.
        undistorted_pt = cv2.undistortPoints(src_pt, self.K, self.D, P=self.K)
        
        # Step 2: Perspective Transform (Homography)
        # Maps the straight pixel to the table surface (mm)
        world_pt = cv2.perspectiveTransform(undistorted_pt, self.H)
        
        # Extract x, y
        x_mm = world_pt[0][0][0]
        y_mm = world_pt[0][0][1]
        
        return x_mm, y_mm