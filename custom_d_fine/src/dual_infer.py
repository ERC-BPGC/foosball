import cv2
import time
import torch
import numpy as np
import threading
from src.infer.torch_model import Torch_model
from src.coordinate_mapper import CoordinateMapper 

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "model_path": "/home/aksh/balltrack/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt",
    "model_name": "l",
    "img_size": (960, 960), 
    "conf_thresh": 0.5,
    "label_to_name": {0: "ball"},
    
    # Camera IDs (Must match your calibration)
    "left_id": 2,   
    "right_id": 3,  
    
    "visualize": True,
    "show_fps": True
}

class ThreadedCamera:
    """
    Reads frames in a separate thread to prevent buffer lag.
    """
    def __init__(self, src=0, width=640, height=480, fps=60):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # Optimize Camera Settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Critical for low latency

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.1)
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
    """
    Calculates the contact point of the ball on the table.
    Using center (cx, cy) is standard for overhead views.
    """
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 
    return cx, cy, x1, y1, x2, y2

def main():
    print("Initializing Project Pinpoint Tracker...")
    
    # 1. Initialize Mappers (The Brain)
    try:
        mapper_l = CoordinateMapper(cam_id=CONFIG["left_id"], side="left")
        mapper_r = CoordinateMapper(cam_id=CONFIG["right_id"], side="right")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load calibration files.\n{e}")
        return

    # 2. Initialize AI Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model on {device}...")
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
    cam_l = ThreadedCamera(CONFIG["left_id"]).start()
    cam_r = ThreadedCamera(CONFIG["right_id"]).start()
    time.sleep(1.0) # Warmup

    print("Tracking Started. Press 'q' to quit.")
    prev_time = 0
    
    try:
        while True:
            # Capture
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()
            
            if not ret_l or frame_l is None: continue
            if not ret_r or frame_r is None: continue

            # Inference on RAW frames (No flipping yet!)
            batch_input = np.stack([frame_l, frame_r])
            results = model(batch_input)
            
            # --- PROCESS LEFT CAMERA ---
            for box, score in zip(results[0]["boxes"], results[0]["scores"]):
                cx, cy, x1, y1, x2, y2 = get_ball_contact_point(box)
                
                # Transform: Pixel (u,v) -> World (mm)
                wx, wy = mapper_l.pixel_to_world(cx, cy)
                
                if CONFIG["visualize"]:
                    cv2.rectangle(frame_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"L: {wx:.0f}, {wy:.0f}"
                    cv2.putText(frame_l, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- PROCESS RIGHT CAMERA ---
            for box, score in zip(results[1]["boxes"], results[1]["scores"]):
                cx, cy, x1, y1, x2, y2 = get_ball_contact_point(box)
                
                # Transform: Pixel (u,v) -> World (mm)
                wx, wy = mapper_r.pixel_to_world(cx, cy)
                
                if CONFIG["visualize"]:
                    cv2.rectangle(frame_r, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"R: {wx:.0f}, {wy:.0f}"
                    cv2.putText(frame_r, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- VISUALIZATION ---
            if CONFIG["visualize"]:
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                # FLIP LEFT CAMERA FOR DISPLAY ONLY
                # This ensures the text is readable but the view is physically oriented correctly for you
                frame_l_vis = cv2.flip(frame_l, 0)
                
                # Stack and show
                display = np.vstack((frame_l_vis, frame_r))
                scale = 0.8
                h, w = display.shape[:2]
                display = cv2.resize(display, (int(w*scale), int(h*scale)))
                
                cv2.putText(display, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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