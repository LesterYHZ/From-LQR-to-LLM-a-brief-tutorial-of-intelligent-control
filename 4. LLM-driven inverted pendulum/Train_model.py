from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- CONFIGURATION ---
max_seq_length = 128 # Physics prompts are short
model_name = "unsloth/Llama-3.2-1B-Instruct" # Perfect for high-speed control

# 1. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters (The "Physics Brain")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Format Data
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

dataset = load_dataset("json", data_files="cartpole_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        # max_steps = 150, # 150 steps is enough for simple regression behavior
        num_train_epochs = 1, # Train on the full dataset once
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "outputs",
        dataloader_num_workers = 0,
    ),
)

print("Starting Training...")
trainer.train()

# 5. Save Model
model.save_pretrained("lora_cartpole_controller")
tokenizer.save_pretrained("lora_cartpole_controller")
print("Model saved locally as 'lora_cartpole_controller'")