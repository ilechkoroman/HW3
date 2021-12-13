import numpy as np


class CursorSystem:
    def __init__(self, F, B, H, Q, R):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.prev_x = None
        self.prev_P = None
        self.control_vector_u = np.array([[0], [0], [0], [0]])

    def init_vars(self, x):
        P = np.zeros(self.Q.shape)
        self.prev_x = (self.F @ x) + (self.B @ self.control_vector_u)
        self.prev_P = (self.F @ P @ self.F.T) + self.Q

    def prediction_step(self):
        # estimation
        # at each step new best estimate made from previous best estimate and
        # a correction for known external influences.
        self.prev_x = (self.F @ self.prev_x) + (self.B @ self.control_vector_u)

        # the same with uncertainty it made from previous uncertainty and
        # environment uncertainty
        self.prev_P = (self.F @ self.prev_P @ self.F.T) + self.Q

    def correction_step(self, z):
        # the overall flow is next we predict, set values at prediction_step()
        # and update in correction_step
        K = self.prev_P @ self.H.T @ np.linalg.inv((self.H @ self.prev_P @ self.H.T) + self.R)
        self.prev_x = self.prev_x + K @ (z - (self.H @ self.prev_x))
        self.prev_P = self.prev_P - K @ self.H @ self.prev_P

    def get_result(self):
        return self.prev_x.astype(int)
