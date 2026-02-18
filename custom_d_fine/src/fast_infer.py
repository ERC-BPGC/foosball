
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from src.infer.torch_model import Torch_model

# ==============================================================================
# CONFIGURATION BLOCK
# ==============================================================================
CONFIG = {
    # Path to your trained model weights (.pt file)
    "model_path": "/home/aksh/balltrack/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt",
    
    # Model architecture type: 'n', 's', 'm', 'l', 'x'
    # MUST match the architecture used during training.
    # If you have a different model (e.g. 's'), change this to 's'.
    "model_name": "l",

    # Input resolution (Height, Width). 
    # Reducing this (e.g. to (640, 640)) significantly increases FPS but may reduce long-range accuracy.
    # Original training was 1280x1280.
    # Intermediate size: 960x960 (Multiple of 32)
    "img_size": (960, 960), 

    # Confidence threshold to accept a detection
    "conf_thresh": 0.5,

    # Class ID to Name mapping (Update if you have more classes)
    "label_to_name": {
        0: "ball"
    },

    # Webcam Device ID (0 is usually default, 1 might be external USB)
    "webcam_id": 1,

    # Visualization
    # Set to False to disable window drawing for maximum performance
    "visualize": True,
    
    # Debug: Draw FPS on screen
    "show_fps": True
}
# ==============================================================================

def main():
    print(f"Initializing D-FINE Real-Time Inference...")
    print(f"Config: {CONFIG}")

    # 1. Initialize Model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        torch_model = Torch_model(
            model_name=CONFIG["model_name"],
            model_path=CONFIG["model_path"],
            n_outputs=len(CONFIG["label_to_name"]),
            input_width=CONFIG["img_size"][1],
            input_height=CONFIG["img_size"][0],
            conf_thresh=CONFIG["conf_thresh"],
            device=device,
            rect=False, # Square input typical for real-time
            half=True   # Use FP16 if possible for speed
        )
        # Warmup to settle CUDA
        torch_model.warmup()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(CONFIG["webcam_id"])
    if not cap.isOpened():
        print(f"Error: Could not open webcam {CONFIG['webcam_id']}")
        return

    # Optimize webcam (optional, might not work on all cameras)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    print("Starting inference loop. Press 'q' or 'Ctrl+C' to exit.")

    prev_time = 0
    curr_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 3. Inference
            # torch_model expects BGR numpy array (standard opencv)
            # Returns list of dicts: [{'boxes': ..., 'scores': ..., 'labels': ...}]
            t0 = time.time()
            results_list = torch_model(frame) 
            t1 = time.time()
            
            # We assume batch size 1
            res = results_list[0]
            boxes = res["boxes"]   # [x1, y1, x2, y2] absolute pixel coords
            scores = res["scores"]
            labels = res["labels"]

            # 4. Process Results
            if len(boxes) > 0:
                # For Control Logic: This is where you'd send coordinates to your controller
                # Example: finding the ball (class 0) with highest score
                best_ball_idx = -1
                best_score = -1
                
                for i, (box, lbl, score) in enumerate(zip(boxes, labels, scores)):
                    if int(lbl) == 0: # Ball
                        if score > best_score:
                            best_score = score
                            best_ball_idx = i
                
                if best_ball_idx != -1:
                    box = boxes[best_ball_idx]
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    # Print coordinates for verification
                    # print(f"Ball detected at: ({center_x:.1f}, {center_y:.1f}) | Score: {best_score:.2f}")

                    if CONFIG["visualize"]:
                        x1, y1, x2, y2 = map(int, box)
                        # Draw Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw Center
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                        # Label
                        cv2.putText(frame, f"Ball {best_score:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 5. Visualization & FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            if CONFIG["show_fps"] and CONFIG["visualize"]:
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # Show inference time latency separately
                # cv2.putText(frame, f"Infer: {(t1-t0)*1000:.1f}ms", (20, 80), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if CONFIG["visualize"]:
                cv2.imshow('D-FINE Real-Time', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
