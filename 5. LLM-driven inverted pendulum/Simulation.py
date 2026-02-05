from unsloth import FastLanguageModel
import torch
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from CustomCartPole import CustomCartPoleEnv

# --- 1. Load Fine-Tuned Model ---
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_cartpole_controller",
    max_seq_length = 128,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# --- 2. Controller Function ---
def llm_policy(state):
    # Prepare Prompt
    # We round to 3 decimals to match training data distribution
    input_str = f"State: [x={state[0]:.3f}, dx={state[1]:.3f}, theta={state[2]:.3f}, dtheta={state[3]:.3f}]"

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a specialized LQR controller. Output the precise continuous force to balance the pendulum.

### Input:
{}

### Response:
"""
    prompt = alpaca_prompt.format(input_str)

    # Inference
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    # Greedy decoding (temperature=0) for deterministic control
    outputs = model.generate(
        **inputs,
        max_new_tokens = 10,
        use_cache = True,
        temperature = 0.0,
        do_sample = False
    )
    text_out = tokenizer.batch_decode(outputs)[0]

    # Parse "Action: -1.234"
    match = re.search(r"Action:\s*(-?[\d\.]+)", text_out)
    if match:
        return float(match.group(1))
    return 0.0 # Safety fallback

# --- 3. Run Simulation & Record Data ---
env = CustomCartPoleEnv(dt=0.1, compute_terminated=False)
state = env.reset()

# Force a specific initial condition to make the plot interesting
# (Tilt the pole 0.1 rad (~5 degrees) to the right)
state_initial = np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32)
env.state = state_initial

history = {
    "time": [],
    "x": [],
    "theta": [],
    "force": []
}

print("Running Simulation with LLM Controller...")
steps = 300
for t in range(steps):
    current_time = t * env.dt

    # Get Action from LLM
    force = llm_policy(state)

    # Record History
    history["time"].append(current_time)
    history["x"].append(state[0])
    history["theta"].append(state[2])
    history["force"].append(force)

    # Step Environment
    next_state, _, done, _ = env.step(force)

    print(f"Time {current_time:.2f}s | Theta: {state[2]:.3f} | Force: {force:.3f}")

    if done:
        print("!!! System Crashed (Out of Bounds) !!!")
        break

    state = next_state

# --- 4. Plot Results ---
print("Generating Plots...")

# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Angle (The main goal is to keep this at 0)
ax1.plot(history["time"], history["theta"], 'b-', linewidth=2, label="Actual Angle")
ax1.axhline(0, color='r', linestyle='--', label="Reference (Upright)")
ax1.set_ylabel("Angle (Radians)")
ax1.set_title("System Output: Pendulum Angle (Theta)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Plot 2: Position (The cart should stay near 0)
ax2.plot(history["time"], history["x"], 'g-', linewidth=2, label="Actual Position")
ax2.axhline(0, color='r', linestyle='--', label="Reference (Center)")
ax2.set_ylabel("Position (Meters)")
ax2.set_title("System Output: Cart Position (x)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# Plot 3: Control Effort (The "Thinking" of the LLM)
ax3.plot(history["time"], history["force"], 'k-', linewidth=1.5, label="LLM Action")
ax3.fill_between(history["time"], history["force"], 0, color='gray', alpha=0.1)
ax3.set_ylabel("Force (Newtons)")
ax3.set_xlabel("Time (Seconds)")
ax3.set_title("Control Effort (LLM Output)")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"Result_dt({env.dt})_initial({state_initial[2]}).png", dpi=300)
plt.show()
