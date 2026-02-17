# Project Status Analysis

**Reference Roadmap**: User's Original Phase 0-3 Plan
**Current State**: Phase 3 (Early Implementation)

---

## Phase 0: Understanding the Context (COMPLETE)
*   **Goal**: Recycle `model.pt` from Quidich football project for Foosball. Setup 2 cameras, use triangulation/calibration.
*   **Status**: **DONE**.
    *   We have the `model.pt`.
    *   We have defined the Table Dimensions (1172x687mm) and "Green Field" context.
    *   We have accepted the 2-Camera Overhead setup.

## Phase 1: Documentation & Understanding (COMPLETE)
*   **Goal**: Understand every line of the D-FINE codebase and document logic.
*   **Status**: **DONE**.
    *   `codebase_documentation.md` generated.
    *   `project_bible.md` generated (contains hardware secrets, driver logic, etc.).
    *   We understand the inference pipeline (`Torch_model`, `postprocess`, etc.).

## Phase 2: Real-time Single Camera Inference (COMPLETE)
*   **Goal**: Strip heavy code, create fast single-cam script (`fast_infer.py`).
*   **Status**: **DONE**.
    *   `src/fast_infer.py`: Created. Removes Hydra/Logging overhead.
    *   `src/infer_with_filter_yolo.py`: Integrated.
    *   **Performance**: Achieved ~25 FPS with Large model on 4060Ti (PyTorch).
    *   **Drivers**: Solved `cv2` issues and V4L2 MJPEG bandwidth issues.

## Phase 3: Dual Camera & 3D/Triangulation (IN PROGRESS)
*   **Goal**:
    1.  Setup 2x PS3 Eye Cameras.
    2.  Calibration (Checkerboard/ArUco).
    3.  Triangulation (Combine feeds for 3D pos).
*   **Status**: **PARTIALLY COMPLETE**.

### What is Done:
1.  **Hardware**: 2x PS3 Eyes set up and working with `pseyepy` / MJPEG drivers.
2.  **Calibration**:
    *   **Solution**: `src/calibrate_field.py`.
    *   **Method**: Used **ArUco Markers (8 total)** instead of checkerboard for robust auto-detection.
    *   **Logic**: Maps Pixels -> Table Millimeters (Homography). This *is* the "Triangulation" for a flat table (z=0). We don't need full 3D stereo reconstruction because the ball is always on the table surface.
3.  **Inference**:
    *   `src/dual_infer.py`: Runs detection on both Left and Right streams.

### What is Left (The "Todo"):
1.  **Fusion Logic**: Currently `dual_infer.py` just *shows* boxes. It needs to:
    *   Convert Box Center $(u,v)$ -> World $(x,y)$ using the Calibration JSON.
    *   Handle the overlap (if both cameras see the ball, which x,y do we trust?).
2.  **Optimization (Critical)**:
    *   Current speed: Running `D-FINE-L` twice (Left + Right) is slow.
    *   **Solution**: Batch Inference or TensorRT (Phase 3.2).
3.  **Tracking**:
    *   Implement Kalman Filter to smooth the noisy $(x,y)$ data and predict trajectory.

---

## Action Plan to Comlete Phase 3
1.  **Refactor Drivers**: Update `dual_infer.py` to use the high-speed `PS3EyeStream` class from `aruco_finall.py`.
2.  **Integrate Calibration**: Update `dual_infer.py` to load `calibration_0.json` and `calibration_2.json` and print **Real World Millimeters** instead of pixels.
3.  **Coordinate Fusion**: Write the logic to merge Left/Right coordinates.
