# Fast Inference Script Documentation

## Overview
`src/fast_infer.py` is a streamlined inference script designed for real-time applications like foosball ball tracking. Unlike the original `infer.py`, it strips away heavy logging, file saving, and complex configuration systems (Hydra) to maximize Frame Per Second (FPS) performance.

## Usage

1.  **Direct Run**:
    ```bash
    python src/fast_infer.py
    ```
    Ensure you are in the `code/custom_d_fine` directory or have the python path set correctly.

## Configuration
The script contains a `CONFIG` dictionary at the very top of the file. You can edit this directly to change parameters.

```python
CONFIG = {
    "model_path": "...",       # Path to your .pt weights
    "model_name": "l",         # Architecture: 'l', 'm', 's', 'n' (Must match weights!)
    "img_size": (1280, 1280),  # Decrease to (640, 640) for higher FPS
    "conf_thresh": 0.5,        # Threshold to filter weak detections
    "visualize": True,         # Set False to disable window (for headless/production)
    "show_fps": True,          # Show performance stats
    "webcam_id": 0             # 0 for default laptop/USB cam
}
```

### Tips for Performance
*   **Resolution**: 1280x1280 is high quality but slow (likely < 15 FPS on typical GPUs). Try changing `img_size` to `(640, 640)`. D-FINE is quite robust and may still detect the ball accurately.
*   **Visualization**: `cv2.imshow` adds latency. For the final robot controller, set `"visualize": False`.

## Output
*   **Visual**: Opens a window showing the live feed with bounding boxes and FPS.
*   **Console**: Prints errors and initialization status.
*   **Code Integration**: Modify the `Process Results` section in the `main` loop to send `(center_x, center_y)` coordinates to your specific hardware controller (e.g., via Serial, UDP, or ROS).

## Model Sizes
You can switch to a lighter model if you have trained weights for them:
*   `n` (Nano): Fastest
*   `s` (Small)
*   `m` (Medium)
*   `l` (Large): Current default, most accurate but slowest.
*   `x` (Extra Large)

**Warning**: If you change `model_name` to 's', you **MUST** update `model_path` to point to a `.pt` file that was trained with the 's' architecture. Mismatched weights will cause a crash.
