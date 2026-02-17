# Robotic Foosball Vision System: The Project Bible

**Current Status**: Phase 3.1 (Functional Baseline)
**Date**: Feb 16, 2026

---

## 1. Project Objective
Build a professional-grade, high-speed vision system for a robotic foosball table.
*   **Goal**: Detect ball position (x, y) at >100 FPS with minimal latency (<10ms).
*   **Hardware Control**: Use Teensy microcontroller to move rods based on vision data (Future Phase).

## 2. Hardware Specification
*   **Cameras**: 2x **PlayStation 3 Eye** (Modified for high FPS).
    *   **Resolution/FPS**: Running at **640x480 @ 60 FPS** (or 320x240 @ 187 FPS using specialized driver).
    *   **Placement**: Overhead, split view (Left Half / Right Half).
*   **Compute**:
    *   **GPU**: NVIDIA RTX 4060Ti (8GB VRAM).
    *   **CPU**: Intel i9 14th Gen.
*   **Table Dimensions**:
    *   **Total Size**: 1172mm x 687mm.
    *   **Center**: (0, 0).
    *   **Left Edge**: x = -586mm.
    *   **Right Edge**: x = +586mm.

## 3. Software Architecture

### A. Driver Layer (`src/driver.py`)
This is the **"Gold Standard"** reference for how we access the cameras.
*   **Class**: `PS3EyeStream`
*   **Backend**: `cv2.CAP_V4L2` (Video4Linux2).
*   **Format**: **MJPEG** (`cv2.VideoWriter_fourcc(*'MJPG')`).
    *   *Why?* Raw YUYV data overloads the USB 2.0 bus when using 2 cameras. MJPEG compresses frames, allowing high FPS on dual streams.
*   **Threading**: Uses `Thread(target=self.update, ...)` to capture frames in the background, preventing I/O blocking.
*   *Note*: Currently `dual_infer.py` uses a local duplicate of this class. Unification is pending.

### B. Coordinate Mapping (`src/coordinate_mapper.py`)
The bridge between Camera Pixels and Real World.
*   **Inputs**:
    *   `config/lens_intrinsics_{side}.json` (Lens Distortion)
    *   `config/calibration_{id}.json` (Homography)
*   **Function**: `pixel_to_world(u, v) -> (x_mm, y_mm)` with correct undistortion.

### C. Calibration (`src/calibrate_field.py`)

Maps Camera Pixels $(u, v)$ to Table Millimeters $(x, y)$.
*   **Method**: ArUco Markers (4x4, Dictionary 50).
*   **Markers Setup**: 8 Markers total, placed on the rail borders.
    *   **Left Camera sees**: 1, 4, 5, 6.
    *   **Right Camera sees**: 2, 3, 7, 8.
*   **Coordinate Map** (Verified in Code):
    ```python
    MARKER_MAP = {
        1: (-586, 343.5),  # Top-Left
        4: (-586, -343.5), # Bottom-Left
        5: (-586, 0),      # Left Goal Center
        6: (0, 343.5),     # Top-Center
        8: (0, -343.5),    # Bottom-Center
        2: (586, 343.5),   # Top-Right
        3: (586, -343.5),  # Bottom-Right
        7: (586, 0)        # Right Goal Center
    }
    ```
*   **Output**: Saves Homography Matrix to `config/calibration_{id}.json`.
*   **Visualization**: Projects a "Yellow Rectangle" representing the ideal field boundaries back onto the video feed to visually confirm accuracy.

### D. Inference Engine (`src/dual_infer.py`)
*   **Model**: **D-FINE** (Large variant).

*   **Weights Path**: `/code/custom_dfine_exp_2x4090_refined_data_2025-07-03/model.pt`.
*   **Logic**:
    1.  Loads D-FINE model onto CUDA.
    2.  Reads Left & Right frames.
    3.  Runs Inference: `model(frame)`.
    4.  **Mapping**: Uses `CoordinateMapper` to convert detection boxes to World Coordinates.
    5.  Visualizes Bounding Boxes + Confidence + $(x, y)$ coordinates.

*   **Current Bottleneck**: Running Large model twice per frame (Left+Right) takes time.
*   **Future Fix (Phase 3.2)**: Export to **TensorRT** (NVIDIA Hardware Acceleration) for 2-4x speedup.

## 4. Key Configuration (`config.yaml`)
*   **Input Size**: `[1280, 1280]` (Training), `[960, 960]` (Inference - resized in script).
*   **Confidence Threshold**: `0.5`.
*   **Classes**: Single class `{0: "ball"}`.

## 5. Critical "Secrets" / Lessons Learned
1.  **Linux USB Bandwidth**: You CANNOT run 2x PS3 Eyes at 640x480 @ 60FPS in standard mode. You *must* force `FourCC='MJPG'`.
2.  **Calibration**: A single camera view (Left/Right) only sees 2 corners of the table. Standard calibration fails with 2 points. We added "Center Markers" (IDs 5,6,7,8) to ensure each camera sees at least 4 points for valid Homography.
3.  **Headless OpenCV**: We faced issues with `opencv-python-headless` breaking `cv2.imshow`. Solution: Installed standard `opencv-python` and confirmed `qt/x11` backend works.

## 6. Directory Structure (Cleaned)
```
~/balltrack/code/custom_d_fine/
├── config.yaml          # Main Config
├── src/
│   ├── driver.py        # Consolidated Camera Class (PS3EyeStream)
│   ├── coordinate_mapper.py # Pixel -> World Mapper
│   ├── calibrate_field.py # Active Calibration Tool
│   ├── dual_infer.py    # Active Inference Tool
│   ├── dual_cam.py      # Dual Camera Viewer (Debug Tool)
│   ├── fast_infer.py    # Backup Single-Cam Tool
│   ├── d_fine/          # Model Architecture
│   └── infer/           # Inference Utils
└── pseyepy/             # Custom Driver Wrapper (for 187 FPS mode)
```

## 7. Immediate Roadmap
1.  **Consolidate Drivers**: Refactor `dual_infer.py` to import `src/driver.py` instead of defining its own camera class.
2.  **Fusion Logic**: Merge the $(x, y)$ outputs from Left and Right cameras into a single tracked object.

3.  **Optimize**: Convert Model to TensorRT (Phase 3.2).
4.  **Track**: Implement Kalman Filter for ball prediction.
