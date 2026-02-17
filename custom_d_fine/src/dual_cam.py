import cv2
import threading
import time
import numpy as np

class FastCapture:
    """
    Spawns a background thread to read from a VideoCapture object continuously.
    This prevents I/O blocking in the main thread and ensures we always get the LATEST frame.
    """
    def __init__(self, src=0, name="Cam"):
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(self.src)
        
        # Try to set high FPS mode for PS3 Eye (if supported by backend)
        # 60 FPS = 60, 75 FPS = 75
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print(f"[{self.name}] Already started.")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"[{self.name}] Started on src={self.src}")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                
            # If camera disconnects, stop
            if not grabbed:
                print(f"[{self.name}] Stopped/Disconnected.")
                self.stop()

    def read(self):
        with self.read_lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
        print(f"[{self.name}] Released.")

class DualCamera:
    """
    Wrapper to manage two cameras simultaneously.
    """
    def __init__(self, left_id=0, right_id=2):
        self.cam_left = FastCapture(left_id, "Left")
        self.cam_right = FastCapture(right_id, "Right")
        
    def start(self):
        self.cam_left.start()
        # Small delay to prevent USB bus contention during startup
        time.sleep(0.5) 
        self.cam_right.start()
        
    def read(self):
        """
        Returns (frame_left, frame_right).
        Either can be None if not available.
        """
        return self.cam_left.read(), self.cam_right.read()
        
    def stop(self):
        self.cam_left.stop()
        self.cam_right.stop()

def main():
    # CONFIG: Camera IDs (Change these to match your system: /dev/video0, /dev/video2, etc.)
    # Often integrated webcam is 0, USB is 2 or 4.
    LEFT_ID = 0
    RIGHT_ID = 2 
    
    print("Initializing Dual Camera System...")
    dual_cam = DualCamera(LEFT_ID, RIGHT_ID)
    dual_cam.start()
    
    print("Press 'q' to quit.")
    
    fps_start = time.time()
    frames = 0
    
    try:
        while True:
            left, right = dual_cam.read()
            
            if left is None and right is None:
                print("Waiting for cameras...")
                time.sleep(0.1)
                continue
            
            # Create a placeholder if one camera is missing
            if left is None:
                left = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(left, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if right is None:
                right = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(right, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Resize for concatenation if needed (ensure same height)
            if left.shape != right.shape:
                right = cv2.resize(right, (left.shape[1], left.shape[0]))
                
            # Concatenate Update
            display = np.hstack((left, right))
            
            # FPS
            frames += 1
            if time.time() - fps_start > 1.0:
                print(f"FPS: {frames}")
                frames = 0
                fps_start = time.time()
            
            cv2.imshow("Dual Camera Stream (Left | Right)", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        dual_cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
