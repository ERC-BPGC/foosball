import torch
import numpy as np
import cv2
from src.infer.torch_model import Torch_model

# Dummy Config for Test
CONFIG = {
    "model_path": "/home/aksh/balltrack/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt",
    "model_name": "l",
    "img_size": (960, 960), 
    "conf_thresh": 0.5,
    "label_to_name": {0: "ball"},
}

def test_batch_inference():
    print("Testing Batch Inference Logic...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
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
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Simulate 2 frames (Left + Right)
    frame_l = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_r = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create batch
    batch_input = np.stack([frame_l, frame_r])
    print(f"Input Batch Shape: {batch_input.shape}")
    
    # Run Inference
    results = model(batch_input)
    
    print(f"Results Count: {len(results)}")
    
    if len(results) == 2:
        print("SUCCESS: Received 2 results (Left + Right).")
        print("Structure Check:")
        print(f"Result 0 Keys: {results[0].keys()}")
    else:
        print(f"FAILURE: Expected 2 results, got {len(results)}")

if __name__ == "__main__":
    test_batch_inference()
