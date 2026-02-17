import cv2
import numpy as np
import json
import time
import argparse
from threading import Thread
from pathlib import Path

# ==============================================================================
# 1. PHYSICAL DIMENSIONS (From Working Code)
# ==============================================================================
FIELD_W = 687 / 2   # 343.5 mm (Half Width)
FIELD_H = 1172 / 2  # 586.0 mm (Half Height)

# Distance from Center of Table to Marker Center (Rail Position)
MARKER_X_DIST = FIELD_W + 50.0  
MARKER_Y_DIST = FIELD_H + 50.0  

# ==============================================================================
# 2. GLOBAL STATE (Smart Defaults)
# ==============================================================================
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
# 3. HELPER CLASSES
# ==============================================================================
class PS3EyeStream:
    def __init__(self, index=0, width=640, height=480, fps=60):
        self.stream = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # Force MJPEG for High FPS
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
            print(f"Error: Camera {index} could not be opened.")

    def start(self):
        if self.is_opened:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed: self.frame = frame
            else:
                self.stopped = True
                break
        self.stream.release()

    def read(self): return self.frame
    def stop(self): self.stopped = True

# ==============================================================================
# 4. CALIBRATION LOGIC
# ==============================================================================
MARKER_MAP = {
    1: (-MARKER_X_DIST, MARKER_Y_DIST), 6: (0, MARKER_Y_DIST), 2: (MARKER_X_DIST, MARKER_Y_DIST),
    5: (-MARKER_X_DIST, 0), 7: (MARKER_X_DIST, 0),
    4: (-MARKER_X_DIST, -MARKER_Y_DIST), 8: (0, -MARKER_Y_DIST), 3: (MARKER_X_DIST, -MARKER_Y_DIST)
}

SIDE_CONFIG = {
    "left":  {"markers": [1, 6, 2, 5, 7], "rect": [1, 2, 7, 5]}, # TL, TR, BR, BL
    "right": {"markers": [5, 7, 4, 8, 3], "rect": [5, 7, 3, 4]}  # TL, TR, BR, BL
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
    """Robust Geometry: Finds the 4th corner if only 3 are visible."""
    found_indices = [i for i, mid in enumerate(corner_ids) if mid in detected_map]
    
    if len(found_indices) != 3: return None, None
    
    p = [None]*4
    for i in found_indices: p[i] = np.array(detected_map[corner_ids[i]])
    miss_idx = [i for i in range(4) if i not in found_indices][0]
    
    pred_pt = None
    # Parallelogram logic
    if miss_idx == 0: pred_pt = p[1] + p[3] - p[2]
    elif miss_idx == 1: pred_pt = p[0] + p[2] - p[3]
    elif miss_idx == 2: pred_pt = p[3] + p[1] - p[0]
    elif miss_idx == 3: pred_pt = p[2] + p[0] - p[1]
    
    return corner_ids[miss_idx], pred_pt

# ==============================================================================
# FIXED SAVE FUNCTION (Place this BEFORE main)
# ==============================================================================
def save_calib(cam_id, side, H, all_offsets):
    try:
        print(f">>> Attempting to save {side} camera settings...")
        
        # 1. Sanitize the offsets (Ensure they are Python ints, not Numpy ints)
        # JSON cannot serialize numpy integers, which causes silent failures.
        side_offsets = {k: int(v) for k, v in all_offsets.items() if k.startswith(side)}
        raw_offsets_safe = {k: int(v) for k, v in all_offsets.items()}

        data = {
            "homography_matrix": H.tolist(),
            "side": side,
            "field_dims": [float(FIELD_W), float(FIELD_H)], # Ensure floats
            "offsets": side_offsets,
            "raw_slider_values": raw_offsets_safe 
        }
        
        # 2. Setup Path
        filename = f"calibration_{cam_id}.json"
        out_dir = Path("config")
        out_dir.mkdir(parents=True, exist_ok=True) # Force create folder
        
        out_path = out_dir / filename
        
        # 3. Write
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"✅ SUCCESS: Saved {side} config to {out_path.absolute()}")
        
    except Exception as e:
        print(f"❌ ERROR SAVING CONFIG: {e}")
        import traceback
        traceback.print_exc()

# ... (Now define process_frame, main, etc.)

def process_frame(frame, side, detector):
    """Main pipeline: Detect -> Extrapolate -> Homography -> Draw Tunable Box"""
    if frame is None: return frame, None

    # 1. Detect Markers (On RAW frame to ensure ArUco validity)
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

    # 2. Extrapolate Missing Corner
    m_id, pred_pt = extrapolate_corner(detected_map, rect_ids)
    if m_id:
        cv2.circle(frame, (int(pred_pt[0]), int(pred_pt[1])), 8, (0,0,255), -1)
        cv2.putText(frame, f"Sim: {m_id}", (int(pred_pt[0])+10, int(pred_pt[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        image_points.append(pred_pt)
        object_points.append(MARKER_MAP[m_id])

    # 3. Compute Homography & Draw
    H_matrix = None
    if len(image_points) >= 4:
        try:
            H_matrix, _ = cv2.findHomography(np.array(image_points, dtype=np.float32), 
                                           np.array(object_points, dtype=np.float32))
            
            # Use Inverse Homography to project the "Tuned Box" onto the screen
            H_inv = np.linalg.inv(H_matrix)
            roi = get_roi_points(side)
            pixel_poly = []
            for pt in roi:
                vec = np.array([pt[0], pt[1], 1.0])
                px = np.dot(H_inv, vec)
                px /= px[2]
                pixel_poly.append([int(px[0]), int(px[1])])
            
            # Draw Yellow ROI
            cv2.polylines(frame, [np.array(pixel_poly, np.int32)], True, (0, 255, 255), 3)
            cv2.putText(frame, "LOCKED", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        except: pass
    
    return frame, H_matrix

# ==============================================================================
# 5. GUI & MAIN LOOP
# ==============================================================================
def nothing(x): pass

def main():
    print("=================================================")
    print("   AIR HOCKEY DUAL-CAMERA CALIBRATION TOOL       ")
    print("=================================================")
    print("Initializing Cameras...")
    print(" -> Left Camera (ID 2): Will be FLIPPED for Display (Top View)")
    print(" -> Right Camera (ID 3): Standard (Bottom View)")

    # 1. Initialize Cameras
    cam_left = PS3EyeStream(index=2).start()
    cam_right = PS3EyeStream(index=3).start()
    time.sleep(1.0) # Warmup

    # 2. Setup ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # 3. Create GUI Windows
    cv2.namedWindow("Tuning Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuning Controls", 600, 500)

    # Helper for sliders (Maps 0-400 to -200 to +200)
    def create_slider(name, default):
        cv2.createTrackbar(name, "Tuning Controls", default + 200, 400, nothing)

    # Create 8 Sliders
    create_slider("L_Top", int(current_offsets["left_top"]))
    create_slider("L_Btm", int(current_offsets["left_bottom"]))
    create_slider("L_Left", int(current_offsets["left_left"]))
    create_slider("L_Right", int(current_offsets["left_right"]))
    
    create_slider("R_Top", int(current_offsets["right_top"]))
    create_slider("R_Btm", int(current_offsets["right_bottom"]))
    create_slider("R_Left", int(current_offsets["right_left"]))
    create_slider("R_Right", int(current_offsets["right_right"]))

    print("\nControls:")
    print("  [Sliders] : Tune edges in real-time")
    print("  [S]       : Save Configuration")
    print("  [Q]       : Quit")

    H_left_final = None
    H_right_final = None

    while True:
        # Read Sliders
        current_offsets["left_top"] = cv2.getTrackbarPos("L_Top", "Tuning Controls") - 200
        current_offsets["left_bottom"] = cv2.getTrackbarPos("L_Btm", "Tuning Controls") - 200
        current_offsets["left_left"] = cv2.getTrackbarPos("L_Left", "Tuning Controls") - 200
        current_offsets["left_right"] = cv2.getTrackbarPos("L_Right", "Tuning Controls") - 200
        
        current_offsets["right_top"] = cv2.getTrackbarPos("R_Top", "Tuning Controls") - 200
        current_offsets["right_bottom"] = cv2.getTrackbarPos("R_Btm", "Tuning Controls") - 200
        current_offsets["right_left"] = cv2.getTrackbarPos("R_Left", "Tuning Controls") - 200
        current_offsets["right_right"] = cv2.getTrackbarPos("R_Right", "Tuning Controls") - 200

        # Read Frames
        f_left = cam_left.read()
        f_right = cam_right.read()

        if f_left is not None and f_right is not None:
            # IMPORTANT: Process frames BEFORE flipping. 
            # ArUco cannot detect markers if the image is mirrored (flipped).
            f_left_proc, hl = process_frame(f_left, "left", detector)
            f_right_proc, hr = process_frame(f_right, "right", detector)
            
            # Save valid homographies
            if hl is not None: H_left_final = hl
            if hr is not None: H_right_final = hr

            # FLIP LEFT CAMERA NOW (Only for Display)
            # This satisfies your requirement to see it flipped, without breaking detection.
            f_left_display = cv2.flip(f_left_proc, 0) 

            # Stack Views: Left (Top) / Right (Bottom)
            combined = np.vstack((f_left_display, f_right_proc))
            
            # Scale down for display
            scale = 0.8
            h, w = combined.shape[:2]
            view = cv2.resize(combined, (int(w*scale), int(h*scale)))

            cv2.imshow("Air Hockey Calibration", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(">>> Saving Configuration...")
            if H_left_final is not None:
                save_calib(2, "left", H_left_final, current_offsets)
            else:
                print("Warning: Left Camera not locked! Not saving left.")
                
            if H_right_final is not None:
                save_calib(3, "right", H_right_final, current_offsets)
            else:
                print("Warning: Right Camera not locked! Not saving right.")

    cam_left.stop()
    cam_right.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()