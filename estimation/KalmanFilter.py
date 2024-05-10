import numpy as np

class KalmanFilter:
    def __init__(self, mu0, P0, action_model, observation_model, Q, R):
        self.mu = mu0
        self.P = P0
        self.Q = Q
        self.R = R

        self.n = self.mu.shape[0]

        self.action_model_ = action_model
        self.observation_model_ = observation_model

    def linearize(self, x, u):
        F, W, H, V = None, None, None, None
        # linearize and get linearized model
        
        return F, W, H, V

    def update(self, u, z):
        # Linearize
        F, W, H, V = self.linearize(self.mu, u)

        # Prediction
        mu = self.action_model_(self.mu, u)
        P = F @ self.P @ F.T + W @ self.Q @ W.T        

        # Correction
        nu = z - self.observation_model_(mu)
        S = H @ P @ H.T + V @ self.R @ V.T
        K = P @ H.T @ np.linalg.inv(S)

        # Update
        self.mu = mu + K @ nu
        self.P = (np.eye(self.n) - K@H) @ P

        return self.mu, self.P