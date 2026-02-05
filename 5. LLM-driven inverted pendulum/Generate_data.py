import numpy as np
import json
import scipy.linalg
from CustomCartPole import CustomCartPoleEnv

# --- 1. SOLVE LQR (Linear Quadratic Regulator) ---
# We linearize the system around the upright fixed point (theta=0)
def solve_lqr(env):
    M, m, l, g = env.mass_cart, env.mass_pole, env.length, env.g
    
    # State Matrix A (Linearized dynamics)
    # x_dot = x_dot
    # x_ddot = ... (linearized)
    # theta_dot = theta_dot
    # theta_ddot = ... (linearized)
    p = M + m
    I = m * l**2 / 3 # Inertia approximation
    denom = I * p + M * m * l**2 
    
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -m**2 * g * l**2 / denom, 0],
        [0, 0, 0, 1],
        [0, 0, m * g * l * p / denom, 0]
    ])
    
    B = np.array([
        [0],
        [(I + m * l**2) / denom],
        [0],
        [-m * l / denom]
    ])

    # Cost Matrices (Penalize angle error heavily)
    Q = np.diag([1.0, 1.0, 100.0, 1.0]) 
    R = np.array([[0.1]]) # Cheap control effort

    # Solve Riccati Equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K[0]

# --- 2. GENERATE DATA ---
def generate(num_samples=60000):
    env = CustomCartPoleEnv()
    K = solve_lqr(env)
    data = []
    
    print("Generating ROBUST expert demonstrations...")
    
    while len(data) < num_samples:
        # 1. Reset to a difficult state (The Key Fix)
        # Randomize angle between -15 and +15 degrees
        # Randomize cart position slightly
        env.reset()
        env.state = np.random.uniform(low=[-0.5, -0.5, -0.2, -0.5], 
                                      high=[0.5, 0.5, 0.2, 0.5])
        
        # Run short episodes (e.g., 20 steps) so we capture the "correction" 
        # but not the boring "staying still" part.
        for _ in range(30): 
            state = env.state
            
            # LQR Control
            force = -np.dot(K, state)
            force = np.clip(force, -10.0, 10.0)
            
            # 2. Add Formatting
            input_str = f"State: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}, {state[3]:.3f}]"
            output_str = f"Action: {force:.3f}"
            
            data.append({
                "instruction": "You are an LQR controller. Balance the pole.",
                "input": input_str,
                "output": output_str
            })
            
            # Step
            env.step(force)
            if len(data) >= num_samples: break

    # Shuffle to prevent "time-series bias" in training
    import random
    random.shuffle(data)

    with open("cartpole_dataset.jsonl", "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(data)} high-variance samples.")

if __name__ == "__main__":
    generate(100000)