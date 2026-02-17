import cv2
import numpy as np
import time
from threading import Thread

# ==============================
# PS3 EYE STREAM CLASS (OPENCV V4L2 VERSION)
# ==============================

class PS3EyeStream:
    def __init__(self, index=0, width=640, height=480, fps=60):
        # Initialize Camera with V4L2 backend (Best for Linux)
        self.stream = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        # ---------------------------------------------------------
        # CRITICAL SETTINGS FOR HIGH FPS ON LINUX
        # ---------------------------------------------------------
        # Force MJPEG video format. Without this, USB 2.0 cannot handle
        # 60fps at 640x480 for two cameras simultaneously.
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        # Optional: Disable Auto Exposure (Values depend on specific driver)
        # self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1=Manual, 3=Auto (V4L2 specific)
        # self.stream.set(cv2.CAP_PROP_EXPOSURE, 120)

        self.stopped = False
        self.frame = None
        self.has_new_frame = False
        self.is_opened = self.stream.isOpened()

        # Read one frame to confirm connection
        if self.is_opened:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
                print(f"Camera {index} initialized: {width}x{height} @ {fps} FPS")
            else:
                self.is_opened = False
                print(f"Camera {index} opened but failed to grab frame.")
        else:
            print(f"Camera {index} failed to open.")

    def start(self):
        if self.is_opened:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.frame = frame
                self.has_new_frame = True
            else:
                self.stopped = True
                break
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        time.sleep(0.1)


# ==============================
# ARUCO SETUP
# ==============================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# ==============================
# MAIN PROGRAM
# ==============================

# Initialize cameras (Check /dev/video* to confirm indices if 0/1 fail)
stream1 = PS3EyeStream(index=2).start()
stream2 = PS3EyeStream(index=3).start() # Try index=2 if this fails

# Allow warm up
time.sleep(1.0)

if not stream1.is_opened or not stream2.is_opened:
    print("Error: Cameras not initialized. Check connections or indices.")
    stream1.stop()
    stream2.stop()
    exit()

print("Running dual PS3 Eye streams with ArUco detection.")
print("Press 'q' to quit.")

# FPS tracking variables
fps_counter1 = 0
fps_counter2 = 0
fps_start1 = time.time()
fps_start2 = time.time()
cap_fps1 = 0
cap_fps2 = 0

display_interval = 2 # Show every Nth frame (Lowered to 2 for smoother UI)
frame_count = 0

while True:
    # Get latest frames (These are already BGR from OpenCV)
    frame1 = stream1.read()
    frame2 = stream2.read()

    # Wait until both cameras have valid frames
    if frame1 is None or frame2 is None:
        continue

    # --- FPS Calculation Camera 1 ---
    fps_counter1 += 1
    if fps_counter1 >= 30: # Update stats every 30 frames
        cap_fps1 = fps_counter1 / (time.time() - fps_start1)
        fps_counter1 = 0
        fps_start1 = time.time()

    # --- FPS Calculation Camera 2 ---
    fps_counter2 += 1
    if fps_counter2 >= 30:
        cap_fps2 = fps_counter2 / (time.time() - fps_start2)
        fps_counter2 = 0
        fps_start2 = time.time()

    # --- Display Throttling ---
    frame_count += 1
    if frame_count % display_interval != 0:
        continue

    # ==============================
    # ARUCO DETECTION CAM 1
    # ==============================
    # No need for BGR conversion, OpenCV is native BGR
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    corners1, ids1, _ = detector.detectMarkers(gray1)

    if ids1 is not None:
        for i, marker_id in enumerate(ids1.flatten()):
            pts = corners1[i][0].astype(int)
            cv2.polylines(frame1, [pts], True, (0,255,0), 2)
            center = np.mean(pts, axis=0).astype(int)
            cv2.circle(frame1, tuple(center), 5, (0,0,255), -1)
            cv2.putText(frame1, f"ID:{marker_id}", (center[0]-20, center[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # ==============================
    # ARUCO DETECTION CAM 2
    # ==============================
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    corners2, ids2, _ = detector.detectMarkers(gray2)

    if ids2 is not None:
        for i, marker_id in enumerate(ids2.flatten()):
            pts = corners2[i][0].astype(int)
            cv2.polylines(frame2, [pts], True, (0,255,0), 2)
            center = np.mean(pts, axis=0).astype(int)
            cv2.circle(frame2, tuple(center), 5, (0,0,255), -1)
            cv2.putText(frame2, f"ID:{marker_id}", (center[0]-20, center[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Display FPS
    cv2.putText(frame1, f'FPS: {cap_fps1:.0f}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame2, f'FPS: {cap_fps2:.0f}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("PS3 Eye 1", frame1)
    cv2.imshow("PS3 Eye 2", frame2)

    # Optional: Print to console less frequently
    if frame_count % 30 == 0:
        print(f"Cam1 FPS: {cap_fps1:.1f} | Cam2 FPS: {cap_fps2:.1f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream1.stop()
stream2.stop()
cv2.destroyAllWindows()