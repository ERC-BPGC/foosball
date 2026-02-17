import cv2
import numpy as np
import json
import argparse
import time
from pathlib import Path
import signal

# ==============================================================================
# CONFIGURATION (Matched to your working snippet)
# ==============================================================================
# NOTE: Ensure these match your physical board exactly!
CHECKERBOARD = (14, 14)  # Inner corners (Rows, Cols)
SQUARE_SIZE_MM = 20      # Size of one square in mm
CAPTURE_DELAY = 1.0      # Seconds between auto-captures
MIN_FRAMES = 15          # Minimum frames required to calibrate

def run_lens_calibration(cam_id, side):
    # ---------------------------------------------------------
    # 1. SETUP
    # ---------------------------------------------------------
    # Prepare object points based on the board configuration
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Setup Camera
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    # Force MJPEG/High Res
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open Camera {cam_id}")
        return

    print(f"============================================================")
    print(f"   AUTO-CALIBRATION: {side.upper()} CAMERA (ID {cam_id})   ")
    print(f"============================================================")
    print(f"Target: {CHECKERBOARD} Grid | {SQUARE_SIZE_MM}mm Squares")
    print("1. Hold the board steady.")
    print(f"2. Auto-capture every {CAPTURE_DELAY}s when pattern is found.")
    print(f"3. Need {MIN_FRAMES}+ frames. Press 'q' to Finish.")

    count = 0
    last_capture_time = time.time()
    
    # Criteria for sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ---------------------------------------------------------
    # 2. CAPTURE LOOP
    # ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Standard detection (Proven to work for you)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_corners:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, ret_corners)

            # AUTO-CAPTURE LOGIC
            time_since_last = time.time() - last_capture_time

            if time_since_last > CAPTURE_DELAY:
                objpoints.append(objp)
                imgpoints.append(corners2)
                count += 1
                last_capture_time = time.time()

                print(f" -> Captured Frame {count}")

                # Visual Flash
                white = np.ones_like(display) * 255
                cv2.addWeighted(display, 0.6, white, 0.4, 0, display)
                cv2.imshow(f'Calibrate {side}', display)
                cv2.waitKey(50)

                # Automatically end capture after 50 frames
                if count >= 50:
                    print("✅ Reached maximum frame count (50). Ending capture.")
                    break
            else:
                # Draw Progress Bar
                bar_width = int((time_since_last / CAPTURE_DELAY) * 640)
                color = (0, 255, 255) if bar_width < 630 else (0, 255, 0)
                cv2.line(display, (0, 470), (bar_width, 470), color, 10)

        # UI Text
        cv2.putText(display, f"Captures: {count}/{MIN_FRAMES}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(f'Calibrate {side}', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if count < 10:
        print("❌ Not enough data to calibrate. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # ---------------------------------------------------------
    # 3. CALCULATE INTRINSICS
    # ---------------------------------------------------------
    print("\n🧮 Calculating Intrinsics... (Please wait)...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print(f"✅ Calibration Complete!")
    print(f"   RMS Error: {ret:.4f} pixels (Lower is better)")

    # Save to JSON
    data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "reprojection_error": ret,
        "image_width": 640,
        "image_height": 480,
        "side": side
    }
    
    out_path = Path(f"config/lens_intrinsics_{side}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"   Saved JSON to: {out_path}")

    # ---------------------------------------------------------
    # 4. VISUAL VERIFICATION (3D Axes)
    # ---------------------------------------------------------
    print("\n👀 VERIFICATION MODE")
    print("   Move the board. X/Y/Z axes should stick to the origin.")
    print("   Press 'q' to Exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_corners:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Solve PnP to get object pose
            success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            
            if success:
                # Draw 3D Axis (Length = 3 squares)
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, SQUARE_SIZE_MM * 3)
                
                # Show Depth
                dist_mm = tvec[2][0]
                cv2.putText(frame, f"Depth: {dist_mm:.0f}mm", (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f'Calibrate {side}', frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# Signal handler for ^C
signal.signal(signal.SIGINT, lambda sig, frame: save_intrinsics_on_interrupt() or exit(0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Camera ID (e.g., 2)")
    parser.add_argument("--side", type=str, required=True, choices=["left", "right"], help="Side")
    args = parser.parse_args()
    
    run_lens_calibration(args.id, args.side)