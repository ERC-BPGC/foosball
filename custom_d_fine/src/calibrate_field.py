import cv2
import time
import torch
import numpy as np
import json
import logging
import threading
from pathlib import Path
from src.infer.torch_model import Torch_model

# ==============================================================================
# 1. CONFIGURATION & PHYSICAL CONSTANTS
# ==============================================================================
CONFIG = {
    "model_path": "/home/aksh/balltrack/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt",
    "model_name": "l",
    "img_size": (960, 960), 
    "conf_thresh": 0.5,
    "label_to_name": {0: "ball"},
    "left_id": 2,   
    "right_id": 3,  
}

# Physical Table Dimensions (mm)
TABLE_L = 1172.0
TABLE_W = 687.0
BALL_DIA = 35.68

# Calculated Math
HALF_L = TABLE_L / 2.0
HALF_W = TABLE_W / 2.0
R = BALL_DIA / 2.0

# When a ball is pushed against a corner, its center is 1 Radius away from the walls.
X_LEFT = -HALF_W + R
X_RIGHT = HALF_W - R
Y_TOP = HALF_L - R         # Attacker side wall
Y_BOTTOM = -HALF_L + R     # Goalie side wall
Y_CENTER = 0.0             # The absolute seam

# The Ground Truth World Coordinates for the Ball's Center
CALIB_POINTS = {
    "left": [
        ("Top-Left Corner", (X_LEFT, Y_TOP)),
        ("Top-Right Corner", (X_RIGHT, Y_TOP)),
        ("Right Wall at Center Line", (X_RIGHT, Y_CENTER)),
        ("Left Wall at Center Line", (X_LEFT, Y_CENTER))
    ],
    "right": [
        ("Left Wall at Center Line", (X_LEFT, Y_CENTER)),
        ("Right Wall at Center Line", (X_RIGHT, Y_CENTER)),
        ("Bottom-Right Corner", (X_RIGHT, Y_BOTTOM)),
        ("Bottom-Left Corner", (X_LEFT, Y_BOTTOM))
    ]
}

# ==============================================================================
# 2. HELPER CLASSES
# ==============================================================================
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480, fps=60):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
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
        with self.read_lock: return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

def save_calibration(side, H_matrix):
    out_dir = Path("config")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{side}.json"
    
    data = {
        "homography_matrix": H_matrix.tolist(),
        "side": side,
        "field_dims": [TABLE_W, TABLE_L],
        "method": "ai_ball_center"
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[SUCCESS] Saved {side} calibration to {out_path}")

# ==============================================================================
# 3. MAIN INTERACTIVE LOOP
# ==============================================================================
def main():
    print("==========================================")
    print("   AI-ASSISTED FOOSBALL CALIBRATION Tool  ")
    print("==========================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading AI Model on {device}...")
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

    print("[INFO] Starting Cameras...")
    cam_l = ThreadedCamera(CONFIG["left_id"]).start()
    cam_r = ThreadedCamera(CONFIG["right_id"]).start()
    time.sleep(1.0) 

    phases = ["left", "right", "done"]
    current_phase_idx = 0
    current_pt_idx = 0
    
    image_points = {"left": [], "right": []}

    print("\n>>> INSTRUCTIONS:")
    print("Place the ball at the requested location and hold it still.")
    print("Press [SPACEBAR] to lock the coordinate.")
    print("Press [R] to restart the current camera.")
    print("Press [Q] to quit immediately.\n")

    try:
        while True:
            phase = phases[current_phase_idx]
            if phase == "done":
                print("\n[INFO] All calibrations completed perfectly!")
                break

            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()
            if not ret_l or not ret_r: continue

            # Align with inference space (Horizontal Flip)
            frame_l_inf = cv2.flip(frame_l, 1)
            frame_r_inf = cv2.flip(frame_r, 1)

            # Inference
            batch_input = np.stack([frame_l_inf, frame_r_inf])
            results = model(batch_input)

            # Active Camera Logic
            if phase == "left":
                active_frame = frame_l_inf
                res = results[0]
            else:
                active_frame = frame_r_inf
                res = results[1]

            h, w = active_frame.shape[:2]
            
            # Find Ball
            box = None
            cx, cy = None, None
            if len(res["boxes"]) > 0:
                box = res["boxes"][0]
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0

            # -------------------------------------------------
            # Display Logic
            # -------------------------------------------------
            # Vertical flip for human viewing
            display = cv2.flip(active_frame, 0) 
            
            target_name, world_pt = CALIB_POINTS[phase][current_pt_idx]
            
            # Draw Box
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                # Transform Y for the vertical display flip
                draw_y1, draw_y2 = h - y2, h - y1
                cv2.rectangle(display, (x1, draw_y1), (x2, draw_y2), (0, 255, 0), 2)
                cv2.circle(display, (int(cx), h - int(cy)), 4, (0, 0, 255), -1)

            # Draw HUD
            cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(display, f"CALIBRATING: {phase.upper()} CAMERA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"TARGET {current_pt_idx+1}/4: {target_name}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            status_text = "Press SPACE to lock" if box is not None else "NO BALL DETECTED"
            status_color = (0, 255, 0) if box is not None else (0, 0, 255)
            cv2.putText(display, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Show previously clicked points
            for pt in image_points[phase]:
                px, py = int(pt[0]), h - int(pt[1])
                cv2.circle(display, (px, py), 6, (255, 0, 0), -1)

            cv2.imshow("AI Calibration", display)

            # -------------------------------------------------
            # Key Handling
            # -------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print(f"[INFO] Resetting {phase.upper()} camera points.")
                image_points[phase] = []
                current_pt_idx = 0
            elif key == ord(' '):
                if box is not None:
                    image_points[phase].append([cx, cy])
                    print(f" -> Locked Point {current_pt_idx+1}: Image({cx:.1f}, {cy:.1f}) mapped to World{world_pt}")
                    
                    current_pt_idx += 1
                    
                    if current_pt_idx == 4:
                        # Compute Homography!
                        src_pts = np.array(image_points[phase], dtype=np.float32)
                        dst_pts = np.array([pt[1] for pt in CALIB_POINTS[phase]], dtype=np.float32)
                        
                        H, _ = cv2.findHomography(src_pts, dst_pts)
                        if H is not None:
                            save_calibration(phase, H)
                            current_phase_idx += 1
                            current_pt_idx = 0
                        else:
                            print("[ERROR] Homography failed. Resetting this side.")
                            image_points[phase] = []
                            current_pt_idx = 0

    except KeyboardInterrupt: pass
    finally:
        cam_l.stop()
        cam_r.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()