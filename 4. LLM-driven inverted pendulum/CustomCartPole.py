import numpy as np
import math

class CustomCartPoleEnv:
    def __init__(self, dt=0.02, compute_terminated=True):
        # Physics Constants
        self.g = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5 # Half-length (center of mass)
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.dt = dt # Seconds between state updates (50Hz)
        self.compute_terminated = compute_terminated

        # Fail thresholds
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.state = None
        self.reset()

    def reset(self):
        # Start slightly random to test controller robustness
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state, dtype=np.float32)

    def step(self, force):
        # Unpack State
        x, x_dot, theta, theta_dot = self.state
        
        # --- LAGRANGIAN DYNAMICS ---
        # Derived from:
        # (M+m)x_ddot + m*l*theta_ddot*cos(theta) - m*l*theta_dot^2*sin(theta) = F
        # m*l*x_ddot*cos(theta) + m*l^2*theta_ddot - m*g*l*sin(theta) = 0
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Solve for angular acceleration (theta_ddot)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.mass_pole * costheta**2 / self.total_mass))
        
        # Solve for linear acceleration (x_ddot)
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler Integration (Update State)
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # Check Termination
        terminated = False
        if self.compute_terminated:
            terminated = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )
        else:
            terminated = False

        return np.array(self.state, dtype=np.float32), 0.0, terminated, {}