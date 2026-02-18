import cv2
import time
import torch
import numpy as np
import threading
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
from src.infer.torch_model import Torch_model
import serial

SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # AI Model
    "model_path": "/home/aksh/balltrack/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt",
    "model_name": "l",
    "img_size": (960, 960), 
    "conf_thresh": 0.5,
    "label_to_name": {0: "ball"},
    
    # Camera IDs (Must match your calibration IDs)
    "left_id": 2,   
    "right_id": 3,  
    
    # Options
    "visualize": True,
    "show_fps": True,
    "config_dir": "config" # Folder where calibration .json files are saved
}

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ==============================================================================
# 1. THE COORDINATE MAPPER (INTEGRATED)
# ==============================================================================
class CoordinateMapper:
    def __init__(self, cam_id: int, side: str, config_dir: str = "config"):
        """
        Robust Coordinate Mapper for Foosball Table.
        """
        self.cam_id = cam_id
        self.side = side.lower()
        self.config_dir = Path(config_dir)

        # Matrices
        self.K = None # Camera Matrix
        self.D = None # Distortion Coeffs
        self.H = None # Homography Matrix

        self._load_configurations()

    def _load_configurations(self):
        lens_path = self.config_dir / f"lens_intrinsics_{self.side}.json"
        field_path = self.config_dir / f"calibration_{self.cam_id}.json"

        # 1. Load Lens Intrinsics (K, D)
        if not lens_path.exists():
            logging.warning(f"[{self.side.upper()}] Lens config missing at {lens_path}. Using identity.")
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
            raise FileNotFoundError(f"CRITICAL: Field calibration missing for {self.side} (ID {self.cam_id}).")
        
        try:
            with open(field_path, "r") as f:
                data = json.load(f)
                self.H = np.array(data["homography_matrix"], dtype=np.float64)
                logging.info(f"[{self.side.upper()}] Mapper Initialized.")
        except Exception as e:
            raise RuntimeError(f"Corrupt Field config for {self.side}: {e}")

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Converts raw pixel (u, v) -> Global World coordinates (x_mm, y_mm).
        """
        if self.H is None: return (0.0, 0.0)

        # 1. Format point for OpenCV
        src_point = np.array([[[u, v]]], dtype=np.float64)

        # 2. Undistort (Fix fisheye)
        undistorted_point = cv2.undistortPoints(src_point, self.K, self.D, P=self.K)

        # 3. Perspective Transform (Homography)
        dst_point = cv2.perspectiveTransform(undistorted_point, self.H)

        return float(dst_point[0][0][0]), float(dst_point[0][0][1])

# ==============================================================================
# 2. CAMERA UTILS
# ==============================================================================
class ThreadedCamera:
    """Reads frames in a separate thread to prevent buffer lag."""
    def __init__(self, src=0, width=640, height=480, fps=60):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # Low Latency Settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.005)
                continue
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                
    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

def get_ball_contact_point(box):
    """Calculates the contact point (center bottom) or center of ball."""
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 
    return cx, cy, x1, y1, x2, y2

# ==============================================================================
# 3. MAIN INFERENCE LOOP
# ==============================================================================
def main():
    print("==========================================")
    print("   PROJECT PINPOINT: DUAL VISION ENGINE   ")
    print("==========================================")
    
    # 1. Initialize Mappers
    try:
        mapper_l = CoordinateMapper(CONFIG["left_id"], "left", CONFIG["config_dir"])
        mapper_r = CoordinateMapper(CONFIG["right_id"], "right", CONFIG["config_dir"])
    except Exception as e:
        print(f"[CRITICAL] Mapper Init Failed: {e}")
        return

    # 2. Initialize AI Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Model on {device}...")
    model = Torch_model(
        model_name=CONFIG["model_name"],
        model_path=CONFIG["model_path"],
        n_outputs=len(CONFIG["label_to_name"]),
        input_width=CONFIG["img_size"][1],
        input_height=CONFIG["img_size"][0],
        conf_thresh=CONFIG["conf_thresh"],
        device=device,
        rect=False,
        half=True
    )
    model.warmup()

    # 3. Start Cameras
    print("[INFO] Starting Cameras...")
    cam_l = ThreadedCamera(CONFIG["left_id"]).start()
    cam_r = ThreadedCamera(CONFIG["right_id"]).start()
    time.sleep(1.0) # Warmup

    # 4. Initialize Serial
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Serial initialized on {SERIAL_PORT} @ {BAUD_RATE}")
    except Exception as e:
        print(f"[WARNING] Could not open serial port: {e}")

    print(">>> TRACKING STARTED. Press 'q' to quit.")
    prev_time = 0
    
    try:
        while True:
            # A. Read Frames
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()
            
            if not ret_l or frame_l is None or not ret_r or frame_r is None:
                continue

            # B. Inference (Batch = 2)
            batch_input = np.stack([frame_l, frame_r])
            results = model(batch_input)
            
            # C. Process Detections
            ball_pos_l = None
            ball_pos_r = None

            # --- Left Camera ---
            if len(results[0]["boxes"]) > 0:
                # Take highest confidence ball
                box = results[0]["boxes"][0]
                cx, cy, x1, y1, x2, y2 = get_ball_contact_point(box)
                wx, wy = mapper_l.pixel_to_world(cx, cy)
                ball_pos_l = (wx, wy)
                
                if CONFIG["visualize"]:
                    cv2.rectangle(frame_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_l, f"L: {wx:.0f},{wy:.0f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Right Camera ---
            if len(results[1]["boxes"]) > 0:
                box = results[1]["boxes"][0]
                cx, cy, x1, y1, x2, y2 = get_ball_contact_point(box)
                wx, wy = mapper_r.pixel_to_world(cx, cy)
                ball_pos_r = (wx, wy)
                
                if CONFIG["visualize"]:
                    cv2.rectangle(frame_r, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_r, f"R: {wx:.0f},{wy:.0f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # D. Data Fusion (The Global Truth)
            final_x, final_y = None, None
            
            if ball_pos_l and ball_pos_r:
                # Average both for stability
                final_x = (ball_pos_l[0] + ball_pos_r[0]) / 2
                final_y = (ball_pos_l[1] + ball_pos_r[1]) / 2
            elif ball_pos_l:
                final_x, final_y = ball_pos_l
            elif ball_pos_r:
                final_x, final_y = ball_pos_r

            # E. Serial Output
            if ser and ser.is_open and final_x is not None:
                try:
                    # Format: "X,Y\n" (e.g., "120,45\n")
                    msg = f"{int(final_x)},{int(final_y)}\n"
                    ser.write(msg.encode('utf-8'))
                except Exception as e:
                    print(f"[ERROR] Serial Write Failed: {e}")

            # F. Visualization
            if CONFIG["visualize"]:
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                # Flip Left for visual consistency
                frame_l_vis = cv2.flip(frame_l, 0)
                
                # Stack
                display = np.vstack((frame_l_vis, frame_r))
                
                # Draw Global Coordinate HUD
                if final_x is not None:
                    hud_text = f"GLOBAL: X={final_x:.0f} mm | Y={final_y:.0f} mm"
                    color = (0, 255, 0) # Green = Tracking
                else:
                    hud_text = "GLOBAL: NO BALL"
                    color = (0, 0, 255) # Red = Lost
                
                # Resize for display
                scale = 0.8
                h, w = display.shape[:2]
                display = cv2.resize(display, (int(w*scale), int(h*scale)))
                
                # Overlay HUD
                cv2.rectangle(display, (0, 0), (w, 60), (0,0,0), -1)
                cv2.putText(display, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display, f"FPS: {fps:.0f}", (int(w*scale)-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Project Pinpoint", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        cam_l.stop()
        cam_r.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()