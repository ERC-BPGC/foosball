import cv2
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
        self.index = index

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
            if not self.stream.isOpened():
                self.stopped = True
                break
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

if __name__ == "__main__":
    # Test the driver
    print("Testing PS3EyeStream...")
    cam = PS3EyeStream(index=0).start()
    
    start_time = time.time()
    frames = 0
    
    try:
        while True:
            frame = cam.read()
            if frame is None: continue
            
            frames += 1
            if time.time() - start_time > 1.0:
                print(f"FPS: {frames}")
                frames = 0
                start_time = time.time()
            
            cv2.imshow("Test", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
