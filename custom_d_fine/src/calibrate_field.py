import cv2
import numpy as np
import json
import time
from threading import Thread
from pathlib import Path

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION
# ==============================================================================
# Field Dimensions (mm)
FIELD_W = 687.0 / 2.0   # 343.5 mm (Half Width)
FIELD_H = 1172.0 / 2.0  # 586.0 mm (Half Height)

# Distance from Center of Table to Marker Center (Rail Position)
MARKER_X_DIST = FIELD_W + 50.0  
MARKER_Y_DIST = FIELD_H + 50.0  

# Default Offsets
current_offsets = {
    # Left Camera (Top Half)
    "left_top": 4, 
    "left_bottom": 41,
    "left_left": 12,    
    "left_right": 16,  
    
    # Right Camera (Bottom Half)
    "right_top": 39,    
    "right_bottom": 6, 
    "right_left": 9, 
    "right_right": 6
}

# ==============================================================================
# 2. HELPER CLASSES
# ==============================================================================
class PS3EyeStream:
    def __init__(self, index=0, width=640, height=480, fps=60):
        self.stream = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # Force MJPEG for High FPS / Low Latency
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        self.stopped = False
        self.frame = None
        self.is_opened = self.stream.isOpened()
        
        if self.is_opened:
            (grabbed, frame) = self.stream.read()
            self.frame = frame if grabbed else np.zeros((height, width, 3), np.uint8)
        else:
            print(f"[ERROR] Camera {index} could not be opened.")

    def start(self):
        if self.is_opened:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed: 
                self.frame = frame
            else:
                self.stopped = True
                break
        self.stream.release()

    def read(self): 
        return self.frame

    def stop(self): 
        self.stopped = True

# ==============================================================================
# 3. MAPPING & UTILITIES
# ==============================================================================
MARKER_MAP = {
    1: (-MARKER_X_DIST, MARKER_Y_DIST), 6: (0, MARKER_Y_DIST), 2: (MARKER_X_DIST, MARKER_Y_DIST),
    5: (-MARKER_X_DIST, 0), 7: (MARKER_X_DIST, 0),
    4: (-MARKER_X_DIST, -MARKER_Y_DIST), 8: (0, -MARKER_Y_DIST), 3: (MARKER_X_DIST, -MARKER_Y_DIST)
}

SIDE_CONFIG = {
    "left":  {"markers": [1, 6, 2, 5, 7], "rect": [1, 2, 7, 5]}, # Top Half
    "right": {"markers": [5, 7, 4, 8, 3], "rect": [5, 7, 3, 4]}  # Bottom Half
}

def get_roi_points(side):
    """Calculates dynamic visual box based on real-time sliders."""
    if side == "left":
        # Top Half (Positive Y)
        y_top = FIELD_H - current_offsets["left_top"]
        y_btm = 0 + current_offsets["left_bottom"]
        x_left = -FIELD_W + current_offsets["left_left"]
        x_right = FIELD_W - current_offsets["left_right"]
        return [(x_left, y_top), (x_right, y_top), (x_right, y_btm), (x_left, y_btm)]
    else:
        # Bottom Half (Negative Y)
        y_top = 0 - current_offsets["right_top"]
        y_btm = -FIELD_H + current_offsets["right_bottom"]
        x_left = -FIELD_W + current_offsets["right_left"]
        x_right = FIELD_W - current_offsets["right_right"]
        return [(x_left, y_top), (x_right, y_top), (x_right, y_btm), (x_left, y_btm)]

def extrapolate_corner(detected_map, corner_ids):
    """Finds the 4th corner if only 3 are visible using parallelogram logic."""
    found_indices = [i for i, mid in enumerate(corner_ids) if mid in detected_map]
    
    if len(found_indices) != 3: return None, None
    
    p = [None]*4
    for i in found_indices: 
        p[i] = np.array(detected_map[corner_ids[i]])
    
    miss_idx = [i for i in range(4) if i not in found_indices][0]
    
    pred_pt = None
    if miss_idx == 0: pred_pt = p[1] + p[3] - p[2]
    elif miss_idx == 1: pred_pt = p[0] + p[2] - p[3]
    elif miss_idx == 2: pred_pt = p[3] + p[1] - p[0]
    elif miss_idx == 3: pred_pt = p[2] + p[0] - p[1]
    
    return corner_ids[miss_idx], pred_pt

def save_calib(cam_id, side, H, all_offsets):
    """Saves calibration data to JSON with explicit type casting."""
    try:
        # 1. Filter and Sanitize Offsets
        side_offsets = {k: int(v) for k, v in all_offsets.items() if k.startswith(side)}
        raw_offsets_safe = {k: int(v) for k, v in all_offsets.items()}

        data = {
            "homography_matrix": H.tolist(),
            "side": side,
            "field_dims": [float(FIELD_W), float(FIELD_H)],
            "offsets": side_offsets,
            "raw_slider_values": raw_offsets_safe 
        }
        
        # 2. Prepare Directory
        out_dir = Path("config")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"calibration_{cam_id}.json"
        
        # 3. Write File
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"[SUCCESS] Saved {side} config to {out_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save {side} config: {e}")
        return False

def process_frame(frame, side, detector):
    """Pipeline: Detect Markers -> Extrapolate -> Homography -> Draw UI"""
    if frame is None: return frame, None

    # Detect Markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    detected_map = {}
    image_points = []
    object_points = []

    valid_ids = SIDE_CONFIG[side]["markers"]
    rect_ids = SIDE_CONFIG[side]["rect"]

    if ids is not None:
        ids = ids.flatten()
        for i, mid in enumerate(ids):
            if mid in valid_ids:
                c = corners[i][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                detected_map[mid] = [cx, cy]
                image_points.append([cx, cy])
                object_points.append(MARKER_MAP[mid])

    # Extrapolate Missing Corner if needed
    m_id, pred_pt = extrapolate_corner(detected_map, rect_ids)
    if m_id:
        pt_int = (int(pred_pt[0]), int(pred_pt[1]))
        cv2.circle(frame, pt_int, 8, (0,0,255), -1)
        cv2.putText(frame, f"Sim: {m_id}", (pt_int[0]+10, pt_int[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        image_points.append(pred_pt)
        object_points.append(MARKER_MAP[m_id])

    # Compute Homography
    H_matrix = None
    if len(image_points) >= 4:
        try:
            # 1. Find Homography
            H_matrix, _ = cv2.findHomography(np.array(image_points, dtype=np.float32), 
                                           np.array(object_points, dtype=np.float32))
            
            # Safety Check: If H_matrix is None (RANSAC failure), skip
            if H_matrix is not None:
                # 2. Project Tunable Box
                H_inv = np.linalg.inv(H_matrix)
                roi = get_roi_points(side)
                pixel_poly = []
                for pt in roi:
                    vec = np.array([pt[0], pt[1], 1.0])
                    px = np.dot(H_inv, vec)
                    
                    # Avoid division by zero
                    scale = px[2] if px[2] != 0 else 1.0
                    px /= scale
                    
                    pixel_poly.append([int(px[0]), int(px[1])])
                
                # 3. Visual Feedback (Draw)
                pts = np.array(pixel_poly, np.int32)
                pts = pts.reshape((-1, 1, 2)) # Ensure correct shape for polylines
                cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
                cv2.putText(frame, "LOCKED", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                
        except Exception as e:
            # PRINT THE ERROR so we know what's wrong!
            print(f"[WARN] Draw Error ({side}): {e}")
            pass
    
    return frame, H_matrix

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def nothing(x): pass

def main():
    print("[INFO] Starting Air Hockey Calibration Tool")
    print("[INFO] Initializing Cameras...")

    # Initialize Cameras
    # ID 2 = Left (Top View), ID 3 = Right (Bottom View)
    cam_left = PS3EyeStream(index=2).start()
    cam_right = PS3EyeStream(index=3).start()
    time.sleep(1.0) # Warmup

    # Setup ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # GUI Setup
    cv2.namedWindow("Tuning Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuning Controls", 600, 500)

    # Sliders
    def create_slider(name, default):
        cv2.createTrackbar(name, "Tuning Controls", default + 200, 400, nothing)

    # Auto-generate sliders based on current_offsets keys
    for key, val in current_offsets.items():
        # Shorten names for UI: "left_top" -> "L_Top"
        name = key.replace("left_", "L_").replace("right_", "R_").capitalize()
        create_slider(name, int(val))

    print("\n[INFO] Controls:")
    print("  [Sliders] : Tune field boundaries")
    print("  [S]       : Save Configuration")
    print("  [Q]       : Quit")

    H_left_final = None
    H_right_final = None

    while True:
        # Update Slider Values (Reading back from UI)
        for key in current_offsets.keys():
            ui_name = key.replace("left_", "L_").replace("right_", "R_").capitalize()
            current_offsets[key] = cv2.getTrackbarPos(ui_name, "Tuning Controls") - 200

        # Read Frames
        f_left = cam_left.read()
        f_right = cam_right.read()

        if f_left is not None and f_right is not None:
            # Process frames BEFORE flipping (ArUco requires original orientation)
            f_left_proc, hl = process_frame(f_left, "left", detector)
            f_right_proc, hr = process_frame(f_right, "right", detector)
            
            # Store valid homographies
            if hl is not None: H_left_final = hl
            if hr is not None: H_right_final = hr

            # Display Logic: Flip Left Camera so it aligns visually with Right Camera
            f_left_display = cv2.flip(f_left_proc, 0) 

            # Stack Views
            combined = np.vstack((f_left_display, f_right_proc))
            
            # Resize for screen
            scale = 0.8
            h, w = combined.shape[:2]
            view = cv2.resize(combined, (int(w*scale), int(h*scale)))

            cv2.imshow("Calibration View", view)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('s'):
            print("\n[INFO] Triggering Save Sequence...")
            
            # Save Left
            if H_left_final is not None:
                save_calib(2, "left", H_left_final, current_offsets)
            else:
                print("[WARN] Left Camera (ID 2): No valid marker lock. Save Aborted.")

            # Save Right
            if H_right_final is not None:
                save_calib(3, "right", H_right_final, current_offsets)
            else:
                print("[WARN] Right Camera (ID 3): No valid marker lock. Save Aborted.")
            print("[INFO] Save Sequence Complete.\n")

    # Cleanup
    cam_left.stop()
    cam_right.stop()
    cv2.destroyAllWindows()
    print("[INFO] Exiting.")

if __name__ == "__main__":
    main()