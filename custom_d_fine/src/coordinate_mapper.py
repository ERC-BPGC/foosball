import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

# Configure clean logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class CoordinateMapper:
    def __init__(self, cam_id: int, side: str, config_dir: str = "config"):
        """
        Robust Coordinate Mapper for Foosball Table.
        
        Args:
            cam_id (int): Camera ID (e.g., 2 or 3)
            side (str): 'left' or 'right'
            config_dir (str): Directory where JSON configs are stored
        """
        self.cam_id = cam_id
        self.side = side.lower()
        self.config_dir = Path(config_dir)

        # Matrices
        self.K = None # Camera Matrix
        self.D = None # Distortion Coeffs
        self.H = None # Homography Matrix

        # Load Configuration immediately
        self._load_configurations()

    def _load_configurations(self):
        """Loads Lens Intrinsics and Homography matrices from JSON."""
        lens_path = self.config_dir / f"lens_intrinsics_{self.side}.json"
        field_path = self.config_dir / f"calibration_{self.cam_id}.json"

        # 1. Load Lens Intrinsics (K, D)
        if not lens_path.exists():
            logging.warning(f"{self.side.upper()} Lens config missing at {lens_path}. Using identity (No Distortion Correction).")
            self.K = np.eye(3, dtype=np.float64)
            self.D = np.zeros(5, dtype=np.float64)
        else:
            try:
                with open(lens_path, "r") as f:
                    data = json.load(f)
                    self.K = np.array(data["camera_matrix"], dtype=np.float64)
                    self.D = np.array(data["dist_coeffs"], dtype=np.float64)
            except Exception as e:
                logging.error(f"Failed to load Lens config: {e}")
                self.K = np.eye(3)
                self.D = np.zeros(5)

        # 2. Load Field Homography (H)
        if not field_path.exists():
            raise FileNotFoundError(f"CRITICAL: Field calibration missing for {self.side} (ID {self.cam_id}). Run calibration first!")
        
        try:
            with open(field_path, "r") as f:
                data = json.load(f)
                self.H = np.array(data["homography_matrix"], dtype=np.float64)
                logging.info(f"[{self.side.upper()}] Mapper Initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Corrupt Field config for {self.side}: {e}")

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Converts raw pixel coordinates (u, v) -> Global World coordinates (x_mm, y_mm).
        
        Logic:
        1. Undistort point (Fix fisheye/lens curvature).
        2. Apply Homography (Map flat 2D point to Global Table Coordinates).
        
        Note: The 'Global Shift' is handled automatically by H if calibration 
              used global marker coordinates.
        """
        if self.H is None:
            return (0.0, 0.0)

        # Step 1: Format point for OpenCV (1, 1, 2)
        # We use float64 for precision
        src_point = np.array([[[u, v]]], dtype=np.float64)

        # Step 2: Undistort
        # P=self.K ensures the output remains in the pixel coordinate space (linearized),
        # rather than normalized camera coordinates.
        undistorted_point = cv2.undistortPoints(src_point, self.K, self.D, P=self.K)

        # Step 3: Perspective Transform (Homography)
        # H maps Linearized Pixels -> Global Millimeters
        dst_point = cv2.perspectiveTransform(undistorted_point, self.H)

        # Extract coordinates
        global_x = float(dst_point[0][0][0])
        global_y = float(dst_point[0][0][1])

        return global_x, global_y