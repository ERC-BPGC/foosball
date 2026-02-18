import cv2
import numpy as np

class FoosballKalman:
    def __init__(self, process_noise=0.03, measurement_noise=0.5):
        """
        State Vector: [x, y, vx, vy]
        Measurement Vector: [x, y]
        """
        self.kf = cv2.KalmanFilter(4, 2) # 4 dynamic params, 2 measured

        # 1. Transition Matrix (F)
        # Defines how state evolves: pos = pos + vel * dt
        # We update 'dt' (delta time) dynamically in the predict step
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)

        # 2. Measurement Matrix (H)
        # We only measure Position (x, y), not Velocity
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 3. Process Noise Covariance (Q)
        # How much we trust the "Physics Model" (Constant Velocity)
        # Low value = Ball moves in straight lines (smooth)
        # High value = Ball changes direction often (erratic)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 4. Measurement Noise Covariance (R)
        # How much noise is in the Camera detection
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # 5. Error Covariance (P) - Initial uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.last_valid_state = None

    def predict(self, dt):
        """
        Predicts the next state based on velocity and time elapsed (dt).
        Must be called EVERY frame, even if no ball is seen.
        """
        # Update dt in the matrix
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        
        predicted = self.kf.predict()
        return predicted.flatten() # Returns [x, y, vx, vy]

    def update(self, x, y):
        """
        Corrects the prediction with actual camera data.
        Call this ONLY when ball is detected.
        """
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kf.correct(measurement)
        self.last_valid_state = corrected.flatten()
        return self.last_valid_state

    def reset(self):
        """Resets filter if ball is lost for too long"""
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.last_valid_state = None