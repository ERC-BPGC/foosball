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
BAUD_RATE = 500000 

# ==============================================================================
# CONFIGURATION & PHYSICAL CONSTANTS
# ==============================================================================
CONFIG = {
    "model_path": "/home/aksh/balltrack/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt",
    "model_name": "l",
    "img_size": (960, 960), 
    "conf_thresh": 0.5,
    "label_to_name": {0: "ball"},
    "left_id": 2,   
    "right_id": 3,  
    "visualize": True,
    "show_fps": True,
    "config_dir": "config"
}

# --- FOOSBALL TABLE KINEMATICS ---
TABLE_W = 343.5 + 20 
TABLE_H = 586.0 + 20

ROD_Y = [150, -120, -420, -600] # Attacker, Midfield, Defender, Goalie
PLAYERS_PER_ROD = [3, 5, 2, 3]
SPACING = [185.0, 127.5, 250.0, 185.0]
LIMITS_STEPS = [2450, 1400, 3600, 2400]

# --- PHYSICS TUNING ---
VELOCITY_DEADBAND = 5.0     
WALL_BOUNCE_DAMP = 0.8      
PREDICTION_CAP_S = 0.35     

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ==============================================================================
# GUI SETUP
# ==============================================================================
def setup_control_panel():
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", 500, 350)
    
    # Trackbars (Name, Window, Default, Max, Callback)
    # Range is 0-200. Subtract 100 in code to get -100 to +100 mm. Default is 100 (which means 0 offset).
    cv2.createTrackbar("X_Off_Att (-100 to 100)", "Control Panel", 100, 200, lambda x: None)
    cv2.createTrackbar("X_Off_Mid (-100 to 100)", "Control Panel", 100, 200, lambda x: None)
    cv2.createTrackbar("X_Off_Def (-100 to 100)", "Control Panel", 100, 200, lambda x: None)
    cv2.createTrackbar("X_Off_Goal (-100 to 100)", "Control Panel", 100, 200, lambda x: None)
    
    cv2.createTrackbar("Fwd_Offset (mm)", "Control Panel", 300, 600, lambda x: None)
    cv2.createTrackbar("Back_Offset (mm)", "Control Panel", 50, 300, lambda x: None)
    
    cv2.createTrackbar("Wall Bounce (1=On)", "Control Panel", 1, 1, lambda x: None)
    cv2.createTrackbar("Kalman (1=On)", "Control Panel", 0, 1, lambda x: None) 

# ==============================================================================
# GAME ENGINE (UNIFIED ZONE STRATEGY)
# ==============================================================================
class GameEngine:
    def __init__(self):
        self.last_linear_steps = [LIMITS_STEPS[0]//2, LIMITS_STEPS[1]//2, LIMITS_STEPS[2]//2, LIMITS_STEPS[3]//2]

    def predict_intersection(self, ball_x: float, ball_y: float, vel_x: float, vel_y: float, rod_y: float, bounce_enabled: bool) -> float:
        """Predicts where the ball will cross a specific rod's Y-axis."""
        if abs(vel_y) < 10.0: 
            return ball_x 
            
        dy = rod_y - ball_y
        time_to_impact = dy / vel_y

        if time_to_impact > PREDICTION_CAP_S: time_to_impact = PREDICTION_CAP_S
        if time_to_impact < 0: return ball_x 

        raw_pred_x = ball_x + (vel_x * time_to_impact)
        
        if bounce_enabled:
            if raw_pred_x > TABLE_W:
                return TABLE_W - ((raw_pred_x - TABLE_W) * WALL_BOUNCE_DAMP)
            elif raw_pred_x < -TABLE_W:
                return -TABLE_W + ((-TABLE_W - raw_pred_x) * WALL_BOUNCE_DAMP)
            
        return raw_pred_x

    def get_motor_commands(self, ball_x: float, ball_y: float, vel_x: float, vel_y: float, params: dict) -> Tuple[list, list, list, list]:
        """Unified Zone Logic Strategy with per-rod X-Offsets."""
        targets_l = [0, 0, 0, 0]
        targets_r = [0, 0, 0, 0]
        active_rods = [False, False, False, False]
        target_xs = [None, None, None, None] # Used for the minimap visualizer
        
        for i in range(4):
            rod_y = ROD_Y[i]
            
            # 1. Apply Live Manual X-Offset specific to this rod
            adj_ball_x = ball_x + params['x_offsets'][i]
            safe_ball_x = max(-TABLE_W, min(adj_ball_x, TABLE_W))
            
            # 2. Check operational zone
            in_zone = (rod_y - params['back_offset']) <= ball_y <= (rod_y + params['fwd_offset'])

            if in_zone:
                active_rods[i] = True
                
                # 3. Predict and Clamp
                target_x = self.predict_intersection(safe_ball_x, ball_y, vel_x, vel_y, rod_y, params['bounce'])
                target_x = max(-TABLE_W, min(target_x, TABLE_W))
                target_xs[i] = target_x # Save for drawing
                
                count = PLAYERS_PER_ROD[i]
                space = SPACING[i]
                max_steps = LIMITS_STEPS[i]
                
                best_steps = -1
                min_dist = float('inf')
                
                for p in range(count):
                    player_offset = (p - (count - 1) / 2.0) * space
                    required_rod_pos = target_x - player_offset
                    
                    percent = (required_rod_pos + TABLE_W) / (2.0 * TABLE_W)
                    req_steps = percent * max_steps
                    
                    if 0 <= req_steps <= max_steps:
                        dist = abs(self.last_linear_steps[i] - req_steps)
                        if dist < min_dist:
                            min_dist = dist
                            best_steps = req_steps
                
                if best_steps < 0:
                    best_steps = 0 if target_x > 0 else max_steps
                    
                targets_l[i] = int(best_steps)
                self.last_linear_steps[i] = int(best_steps)
                targets_r[i] = 1 
                
            else:
                targets_l[i] = self.last_linear_steps[i]
                targets_r[i] = 0 
                
        return targets_l, targets_r, active_rods, target_xs

# ==============================================================================
# COORDINATE MAPPER & CAMERA CLASSES
# ==============================================================================
class CoordinateMapper:
    def __init__(self, cam_id: int, side: str, config_dir: str = "config"):
        self.cam_id = cam_id
        self.side = side.lower()
        self.config_dir = Path(config_dir)
        self.K, self.D, self.H = None, None, None
        self._load_configurations()

    def _load_configurations(self):
        lens_path = self.config_dir / f"lens_intrinsics_{self.side}.json"
        field_path = self.config_dir / f"calibration_{self.side}.json"

        if not lens_path.exists():
            self.K = np.eye(3, dtype=np.float64); self.D = np.zeros(5, dtype=np.float64)
        else:
            with open(lens_path, "r") as f: d = json.load(f)
            self.K = np.array(d["camera_matrix"], dtype=np.float64)
            self.D = np.array(d["dist_coeffs"], dtype=np.float64)

        if not field_path.exists():
            raise FileNotFoundError(f"CRITICAL: Field calibration missing for {self.side}")
        
        with open(field_path, "r") as f: d = json.load(f)
        self.H = np.array(d["homography_matrix"], dtype=np.float64)

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        if self.H is None: return (0.0, 0.0)
        src_point = np.array([[[u, v]]], dtype=np.float64)
        undistorted_point = cv2.undistortPoints(src_point, self.K, self.D, P=self.K)
        dst_point = cv2.perspectiveTransform(undistorted_point, self.H)
        return float(dst_point[0][0][0]), float(dst_point[0][0][1])

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

def get_ball_contact_point(box):
    x1, y1, x2, y2 = map(int, box)
    return (x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2

def draw_radar_minimap(display, ball_x, ball_y, target_xs, params, active_rods):
    """Draws a 2D top-down mini-map on the screen for live visual debugging."""
    scale = 0.20
    map_w, map_h = int(TABLE_W * 2 * scale), int(TABLE_H * 2 * scale)
    pad = 20
    
    # Position map in top right corner
    sx = display.shape[1] - map_w - pad
    sy = pad
    
    # Draw Table Background
    cv2.rectangle(display, (sx, sy), (sx + map_w, sy + map_h), (30, 60, 30), -1)
    cv2.rectangle(display, (sx, sy), (sx + map_w, sy + map_h), (200, 200, 200), 2)
    
    def to_map(wx, wy):
        mx = sx + int((wx + TABLE_W) * scale)
        my = sy + int((TABLE_H - wy) * scale) # Flip Y so Opponent (+Y) is at top
        return mx, my

    # Draw Rods, Zones, and Targets
    for i, rod_y in enumerate(ROD_Y):
        # 1. Zone Bracket
        zx1, zy1 = to_map(-TABLE_W, rod_y + params['fwd_offset'])
        zx2, zy2 = to_map(TABLE_W, rod_y - params['back_offset'])
        
        if active_rods[i]:
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (0, 255, 0), cv2.FILLED)
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (255, 255, 255), 1)
        else:
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (100, 100, 100), 1)
        
        # 2. Solid Rod Line
        rx1, ry = to_map(-TABLE_W, rod_y)
        rx2, _ = to_map(TABLE_W, rod_y)
        cv2.line(display, (rx1, ry), (rx2, ry), (255, 255, 255), 2)

        # 3. Draw Target Marker (where this specific rod is aiming)
        if target_xs is not None and target_xs[i] is not None:
            tx, ty = to_map(target_xs[i], rod_y)
            cv2.circle(display, (tx, ty), 3, (0, 255, 255), -1) # Yellow Dot for Aim

    # Draw Ball (Raw Camera Input)
    if ball_x is not None:
        bx, by = to_map(ball_x, ball_y)
        cv2.circle(display, (bx, by), 4, (0, 0, 255), -1) # Red Dot

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    print("==========================================")
    print("   PROJECT PINPOINT: PHASE 1 (LIVE TUNING)    ")
    print("==========================================")
    
    try:
        mapper_l = CoordinateMapper(CONFIG["left_id"], "left", CONFIG["config_dir"])
        mapper_r = CoordinateMapper(CONFIG["right_id"], "right", CONFIG["config_dir"])
    except Exception as e:
        print(f"[CRITICAL] Mapper Init Failed: {e}"); return

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

    print("[INFO] Starting Cameras...")
    cam_l = ThreadedCamera(CONFIG["left_id"]).start()
    cam_r = ThreadedCamera(CONFIG["right_id"]).start()
    time.sleep(1.0) 

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Serial initialized on {SERIAL_PORT} @ {BAUD_RATE}")
    except Exception as e:
        print(f"[WARNING] Could not open serial port: {e}")

    engine = GameEngine()
    
    if CONFIG["visualize"]:
        setup_control_panel()

    print(">>> TRACKING STARTED. Press 'q' to quit.")
    
    prev_time = 0
    prev_ball_pos = None
    
    try:
        while True:
            # 0. READ GUI SLIDERS
            params = {
                'x_offsets': [0, 0, 0, 0],
                'fwd_offset': 300, 'back_offset': 50, 
                'bounce': True, 'kalman': False
            }
            if CONFIG["visualize"]:
                params['x_offsets'] = [
                    cv2.getTrackbarPos("X_Off_Att (-100 to 100)", "Control Panel") - 100,
                    cv2.getTrackbarPos("X_Off_Mid (-100 to 100)", "Control Panel") - 100,
                    cv2.getTrackbarPos("X_Off_Def (-100 to 100)", "Control Panel") - 100,
                    cv2.getTrackbarPos("X_Off_Goal (-100 to 100)", "Control Panel") - 100
                ]
                params['fwd_offset'] = cv2.getTrackbarPos("Fwd_Offset (mm)", "Control Panel")
                params['back_offset'] = cv2.getTrackbarPos("Back_Offset (mm)", "Control Panel")
                params['bounce'] = bool(cv2.getTrackbarPos("Wall Bounce (1=On)", "Control Panel"))
                params['kalman'] = bool(cv2.getTrackbarPos("Kalman (1=On)", "Control Panel"))

            # 1. TIME & PERCEPTION
            curr_time = time.time()
            dt = curr_time - prev_time if prev_time > 0 else 0.016
            prev_time = curr_time

            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()
            
            frame_l = cv2.flip(frame_l, 1)
            frame_r = cv2.flip(frame_r, 1)  
            
            if not ret_l or frame_l is None or not ret_r or frame_r is None: continue

            batch_input = np.stack([frame_l, frame_r])
            results = model(batch_input)
            
            ball_pos_l = None
            ball_pos_r = None
            h_l, w_l = frame_l.shape[:2]
            h_r, w_r = frame_r.shape[:2]

            det_l = None
            if len(results[0]["boxes"]) > 0:
                box = results[0]["boxes"][0]
                cx, cy, x1, y1, x2, y2 = get_ball_contact_point(box)
                wx, wy = mapper_l.pixel_to_world(cx, cy)

                if abs(wx) <= TABLE_W and abs(wy) <= TABLE_H:
                    ball_pos_l = (wx, wy)
                    det_l = (wx, wy, x1, y1, x2, y2)
                    cx_l, cy_l = cx, cy

            det_r = None
            if len(results[1]["boxes"]) > 0:
                box = results[1]["boxes"][0]
                cx, cy, x1, y1, x2, y2 = get_ball_contact_point(box)
                wx, wy = mapper_r.pixel_to_world(cx, cy)
                
                if abs(wx) <= TABLE_W and abs(wy) <= TABLE_H:
                    ball_pos_r = (wx, wy)
                    det_r = (wx, wy, x1, y1, x2, y2)
                    cx_r, cy_r = cx, cy

            # 2. DATA FUSION
            final_x, final_y = None, None
            if ball_pos_l and ball_pos_r:
                center_l = np.array([w_l / 2, h_l / 2])
                weight_l = 1.0 / (np.linalg.norm(np.array([cx_l, cy_l]) - center_l) + 1e-5)
                center_r = np.array([w_r / 2, h_r / 2])
                weight_r = 1.0 / (np.linalg.norm(np.array([cx_r, cy_r]) - center_r) + 1e-5)

                total_weight = weight_l + weight_r
                final_x = (ball_pos_l[0] * weight_l + ball_pos_r[0] * weight_r) / total_weight
                final_y = (ball_pos_l[1] * weight_l + ball_pos_r[1] * weight_r) / total_weight
            elif ball_pos_l:
                final_x, final_y = ball_pos_l
            elif ball_pos_r:
                final_x, final_y = ball_pos_r

            # 3. KINEMATICS & SERIAL TRANSMISSION
            msg = None
            vel_x, vel_y = 0.0, 0.0
            active_rods = [False, False, False, False]
            target_xs = None
            
            if final_x is not None:
                if prev_ball_pos is not None and dt > 0.005:
                    raw_vx = (final_x - prev_ball_pos[0]) / dt
                    raw_vy = (final_y - prev_ball_pos[1]) / dt
                    
                    vel_x = raw_vx if abs(raw_vx) > VELOCITY_DEADBAND else 0.0
                    vel_y = raw_vy if abs(raw_vy) > VELOCITY_DEADBAND else 0.0
                    
                prev_ball_pos = (final_x, final_y)

                # Send Live Parameters to Game Engine
                target_l, target_r, active_rods, target_xs = engine.get_motor_commands(final_x, final_y, vel_x, vel_y, params)

                msg = f"<{target_l[0]},{target_r[0]},{target_l[1]},{target_r[1]},{target_l[2]},{target_r[2]},{target_l[3]},{target_r[3]}>\n"
                
                if ser and ser.is_open:
                    try:
                        ser.write(msg.encode('utf-8'))
                    except Exception as e:
                        print(f"[ERROR] Serial Write Failed: {e}")

            # 4. VISUALIZATION & TELEMETRY
            if CONFIG["visualize"]:
                frame_l = cv2.flip(frame_l, 0)
                frame_r = cv2.flip(frame_r, 0)

                if det_l:
                    wx, wy, x1, y1, x2, y2 = det_l
                    y1_f, y2_f = h_l - y2, h_l - y1
                    cv2.rectangle(frame_l, (x1, y1_f), (x2, y2_f), (0, 255, 0), 2)
                    cv2.putText(frame_l, f"L: {wx:.0f},{wy:.0f}", (x1, y1_f-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if det_r:
                    wx, wy, x1, y1, x2, y2 = det_r
                    y1_f, y2_f = h_r - y2, h_r - y1
                    cv2.rectangle(frame_r, (x1, y1_f), (x2, y2_f), (0, 255, 0), 2)
                    cv2.putText(frame_r, f"R: {wx:.0f},{wy:.0f}", (x1, y1_f-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                fps = 1 / dt if dt > 0 else 0
                display = np.vstack((frame_l, frame_r))
                
                scale = 0.8
                new_w, new_h = int(display.shape[1]*scale), int(display.shape[0]*scale)
                display = cv2.resize(display, (new_w, new_h))
                
                # --- DRAW THE MINI-MAP RADAR ---
                draw_radar_minimap(display, final_x, final_y, target_xs, params, active_rods)

                if final_x is not None:
                    hud_text_1 = f"BALL POS: X={final_x:.0f}mm, Y={final_y:.0f}mm | VelY: {vel_y:.0f} mm/s"
                    color = (50, 255, 50) 
                else:
                    hud_text_1 = "BALL POS: [LOST]"
                    color = (50, 50, 255) 
                
                if msg is not None:
                    if "1" in msg:
                        hud_text_2 = f"SERIAL OUT: {msg.strip()}   >>> KICKING! <<<"
                    else:
                        hud_text_2 = f"SERIAL OUT: {msg.strip()}"
                else:
                    hud_text_2 = "SERIAL OUT: [IDLE]"

                cv2.rectangle(display, (0, 0), (new_w, 85), (20, 20, 20), -1) 
                cv2.putText(display, hud_text_1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, hud_text_2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2) 
                cv2.putText(display, f"FPS: {fps:.0f}", (new_w - 300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow("Project Pinpoint DEBUG", display)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
    except KeyboardInterrupt: pass
    finally:
        cam_l.stop()
        cam_r.stop()
        if ser: ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()