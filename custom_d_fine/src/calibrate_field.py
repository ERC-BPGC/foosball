import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Camera IDs — must match dual_infer.py
LEFT_CAM_ID = 3    # Top in display (left side of table)
RIGHT_CAM_ID = 2   # Bottom in display (right side of table)

# Field Dimensions (mm) — half-values
HALF_W = 344.25    # Half width of playing field
FIELD_H = 586.0   # Half height (one camera covers this much)

# World coordinates for clicked corners (TL, TR, BR, BL order)
# Origin (0,0) is at the seam between the two cameras:
#   - Bottom-right midpoint of top rectangle
#   - Top midpoint of bottom rectangle
WORLD_POINTS = {
    "left": np.array([   # Top camera covers positive Y
        [-HALF_W,  FIELD_H],   # TL
        [ HALF_W,  FIELD_H],   # TR
        [ HALF_W,  0.0    ],   # BR
        [-HALF_W,  0.0    ],   # BL
    ], dtype=np.float32),
    "right": np.array([  # Bottom camera covers negative Y
        [-HALF_W,  0.0     ],  # TL
        [ HALF_W,  0.0     ],  # TR
        [ HALF_W, -FIELD_H ],  # BR
        [-HALF_W, -FIELD_H ],  # BL
    ], dtype=np.float32),
}

CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
CORNER_COLORS = [(0, 200, 255), (0, 255, 0), (255, 0, 100), (255, 100, 0)]

# ==============================================================================
# THREADED CAMERA (same as dual_infer.py)
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
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
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

# ==============================================================================
# CLICK HANDLER
# ==============================================================================
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """Collect clicked points (up to 4)."""
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"  [{len(clicked_points)}/4] Clicked: ({x}, {y}) — {CORNER_LABELS[len(clicked_points)-1]}")

# ==============================================================================
# APPLY SAME FLIPS AS dual_infer.py
# ==============================================================================
def apply_flips(frame):
    """Apply the same flips as dual_infer.py: horizontal on read, vertical for display."""
    # Step 1: Horizontal flip (same as dual_infer.py does on read)
    frame_h = cv2.flip(frame, 1)
    # Step 2: Vertical flip (same as dual_infer.py does before display)
    frame_hv = cv2.flip(frame_h, 0)
    return frame_h, frame_hv

# ==============================================================================
# CALIBRATE ONE CAMERA
# ==============================================================================
def calibrate_camera(cam, cam_id, side, win_name):
    """
    Interactive calibration for one camera.
    
    The user clicks 4 corners on the DISPLAY frame (which has both H+V flips).
    We un-do the vertical flip to get coordinates in INFERENCE space (H-flip only),
    since dual_infer.py runs inference on the H-flipped frame.
    
    Returns the homography matrix or None.
    """
    global clicked_points
    clicked_points = []

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    print(f"\n{'='*50}")
    print(f"  CALIBRATING: {side.upper()} CAMERA (ID {cam_id})")
    print(f"{'='*50}")
    print(f"  Click the 4 corners of the playing field in order:")
    print(f"    1. Top-Left    2. Top-Right")
    print(f"    3. Bottom-Right  4. Bottom-Left")
    print(f"  [R] = Redo  |  [S] = Save  |  [Q] = Skip\n")

    H_matrix = None

    while True:
        ret, raw_frame = cam.read()
        if not ret or raw_frame is None:
            continue

        # Apply same flips as dual_infer.py
        frame_infer, frame_display = apply_flips(raw_frame)
        h, w = frame_display.shape[:2]

        # Draw existing clicked points
        vis = frame_display.copy()
        for i, pt in enumerate(clicked_points):
            color = CORNER_COLORS[i]
            cv2.circle(vis, pt, 8, color, -1)
            cv2.putText(vis, CORNER_LABELS[i], (pt[0]+12, pt[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Connect consecutive corners
            if i > 0:
                cv2.line(vis, clicked_points[i-1], pt, (255, 255, 255), 2)

        # Close polygon if all 4 clicked
        if len(clicked_points) == 4:
            cv2.line(vis, clicked_points[3], clicked_points[0], (255, 255, 255), 2)
            # Fill polygon semi-transparent
            overlay = vis.copy()
            pts_arr = np.array(clicked_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 255, 80))
            vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

        # HUD
        if len(clicked_points) < 4:
            next_corner = CORNER_LABELS[len(clicked_points)]
            hud = f"{side.upper()} (ID {cam_id}) | Click: {next_corner} ({len(clicked_points)}/4)"
            hud_color = (0, 200, 255)
        else:
            hud = f"{side.upper()} (ID {cam_id}) | DONE! [S]=Save  [R]=Redo  [Q]=Skip"
            hud_color = (0, 255, 0)

        cv2.rectangle(vis, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(vis, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            # Redo
            clicked_points = []
            H_matrix = None
            print(f"  [REDO] Cleared points for {side}. Click again.")

        elif key == ord('s') and len(clicked_points) == 4:
            # Convert display clicks → inference-space coordinates
            # Display has H+V flip. Inference space has only H flip.
            # So we need to undo the V flip: y_infer = h - y_display
            infer_points = []
            for (px, py) in clicked_points:
                infer_points.append([px, h - py])
            
            pixel_pts = np.array(infer_points, dtype=np.float32)
            world_pts = WORLD_POINTS[side]

            H_matrix, status = cv2.findHomography(pixel_pts, world_pts)
            if H_matrix is not None:
                print(f"  [OK] Homography computed for {side}.")
                # Save to file
                save_calibration(cam_id, side, H_matrix)
            else:
                print(f"  [ERROR] Homography computation failed! Try again.")
                clicked_points = []
                continue
            break

        elif key == ord('q'):
            print(f"  [SKIP] Skipping {side} calibration.")
            break

    cv2.destroyWindow(win_name)
    return H_matrix

# ==============================================================================
# SAVE CALIBRATION
# ==============================================================================
def save_calibration(cam_id, side, H_matrix):
    """Save homography to config/calibration_{cam_id}.json."""
    out_dir = Path("config")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{cam_id}.json"

    data = {
        "homography_matrix": H_matrix.tolist(),
        "side": side,
        "field_dims": [float(HALF_W), float(FIELD_H)],
        "method": "manual_corners",
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"  [SAVED] {out_path}")

# ==============================================================================
# VERIFICATION: test a few points after calibration
# ==============================================================================
def verify_calibration(cam, cam_id, side, H_matrix, win_name):
    """
    After calibration, let the user click anywhere on the feed
    and see the mapped world coordinate. Press Q to exit.
    """
    global clicked_points
    clicked_points = []

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    print(f"\n  [VERIFY] Click anywhere to see world coordinates. [Q] = Done.")

    while True:
        ret, raw_frame = cam.read()
        if not ret or raw_frame is None:
            continue

        frame_infer, frame_display = apply_flips(raw_frame)
        h, w = frame_display.shape[:2]
        vis = frame_display.copy()

        # Draw all test clicks with their mapped coordinates
        for pt in clicked_points:
            px, py = pt
            # Convert to inference space
            infer_x, infer_y = px, h - py
            # Apply homography
            src = np.array([[[infer_x, infer_y]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, H_matrix)
            wx, wy = dst[0][0][0], dst[0][0][1]

            cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"({wx:.0f}, {wy:.0f})", (px+10, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # HUD
        cv2.rectangle(vis, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(vis, f"VERIFY {side.upper()} | Click to test | [Q]=Done",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyWindow(win_name)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 50)
    print("  MANUAL FIELD CALIBRATION TOOL")
    print("=" * 50)
    print(f"  Left Camera  (top):    ID {LEFT_CAM_ID}")
    print(f"  Right Camera (bottom): ID {RIGHT_CAM_ID}")
    print(f"  Field: {HALF_W*2:.0f} x {FIELD_H*2:.0f} mm")
    print(f"  Origin (0,0) at camera seam")
    print()

    # Start cameras
    print("[INFO] Starting cameras...")
    cam_l = ThreadedCamera(LEFT_CAM_ID).start()
    cam_r = ThreadedCamera(RIGHT_CAM_ID).start()
    time.sleep(1.0)

    # Calibrate left (top) camera
    H_left = calibrate_camera(cam_l, LEFT_CAM_ID, "left", "Calibrate LEFT (Top)")

    # Verify left
    if H_left is not None:
        verify_calibration(cam_l, LEFT_CAM_ID, "left", H_left, "Verify LEFT")

    # Calibrate right (bottom) camera
    H_right = calibrate_camera(cam_r, RIGHT_CAM_ID, "right", "Calibrate RIGHT (Bottom)")

    # Verify right
    if H_right is not None:
        verify_calibration(cam_r, RIGHT_CAM_ID, "right", H_right, "Verify RIGHT")

    # Summary
    print("\n" + "=" * 50)
    print("  CALIBRATION COMPLETE")
    print("=" * 50)
    if H_left is not None:
        print(f"  ✓ Left  (ID {LEFT_CAM_ID}) → config/calibration_{LEFT_CAM_ID}.json")
    else:
        print(f"  ✗ Left  (ID {LEFT_CAM_ID}) — skipped")
    if H_right is not None:
        print(f"  ✓ Right (ID {RIGHT_CAM_ID}) → config/calibration_{RIGHT_CAM_ID}.json")
    else:
        print(f"  ✗ Right (ID {RIGHT_CAM_ID}) — skipped")

    # Cleanup
    cam_l.stop()
    cam_r.stop()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()