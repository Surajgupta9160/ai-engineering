# 16 — Fine-Tuning: Teaching Models Your Specific Style and Domain

---

## 1. What Is Fine-Tuning?

Fine-tuning is continuing the training of a pre-trained model on your own specific data. Think of it as the "intern to specialist" analogy:

```
PRE-TRAINED MODEL:         FINE-TUNED MODEL:
  "Brilliant generalist"     "Domain specialist"
  Knows everything           Knows YOUR domain deeply
  Generic responses          Responses in YOUR style
  Verbose explanations       Concise, formatted correctly
  May need long prompts      Works with short prompts
  Like a new hire            Like a 6-month employee
```

### What Fine-Tuning Changes

```
Model weights before fine-tuning:
  [weight_1: 0.234, weight_2: -0.891, ..., weight_N: 0.445]

Your training examples:
  {"prompt": "Summarize this...", "response": "Key points: 1. 2. 3."}
  {"prompt": "Write subject line", "response": "Subject: [Action] by [Date]"}

Model weights after fine-tuning:
  [weight_1: 0.241, weight_2: -0.887, ..., weight_N: 0.451]
  ← Slightly adjusted to better produce your expected outputs

Result: Model "knows" your preferred style without being told each time.
```

---

## 2. When to Fine-Tune

```
DECISION FLOWCHART:
───────────────────

Can prompting solve the problem?
  YES → Use prompting (cheapest, fastest, most flexible)
  NO  → Continue

Can RAG solve the problem (private/recent data)?
  YES → Use RAG (no training needed, data stays fresh)
  NO  → Continue

Is the problem one of these?
  □ Need consistent tone/style across ALL responses
  □ Domain-specific vocabulary or abbreviations
  □ Want to reduce prompt length (cost savings at scale)
  □ Need behavior that prompting can't achieve
  □ Privacy concern (don't want examples in every prompt)
  YES → Fine-tune!
  NO  → Reconsider if fine-tuning is really needed

FINE-TUNING IS GOOD FOR:            FINE-TUNING IS NOT FOR:
✓ Custom response style/format       ✗ Adding factual knowledge (use RAG)
✓ Domain jargon/terminology          ✗ One-time tasks
✓ Reducing prompt length             ✗ Tasks that change frequently
✓ Brand voice consistency            ✗ Real-time or dynamic information
✓ Specialised task performance       ✗ When prompting already works
```

---

## 3. Types of Fine-Tuning

### Full Fine-Tuning
```
ALL model weights are updated during training.
Cost: Very high (same scale as original training)
Quality: Maximum
Risk: Catastrophic forgetting (may lose general knowledge)
Use case: Almost never for AI engineers (done by model providers)
```

### LoRA (Low-Rank Adaptation)

```
INTUITION:
  Instead of updating all 7 billion weights,
  add tiny "adapter" layers and only update those.

  Original weight matrix W (large, frozen):
  ┌─────────────────────────────┐
  │ 0.21  -0.08   0.64   0.15  │  ← FROZEN (not updated)
  │ 0.55   0.33  -0.22   0.89  │
  │ ...                         │
  └─────────────────────────────┘
                +
  LoRA adapters (tiny, trainable):
  ┌──────┐   ┌──────────────────────┐
  │  A   │ × │         B            │  ← Only A and B are trained
  │ d×r  │   │         r×k          │
  └──────┘   └──────────────────────┘
  (d=in, k=out, r=rank — much smaller!)

  W_new = W_frozen + scaling × (A × B)

NUMBERS (Llama 3.1 8B):
  Full fine-tuning: 8,000,000,000 trainable params
  LoRA (rank=16):      41,943,040 trainable params (0.5%!)
  Memory saved: ~95% less GPU RAM needed
```

### QLoRA (Quantized LoRA)

```
QLoRA = Quantization + LoRA

Step 1: Load the base model in 4-bit precision (saves memory)
  FP32: Each weight = 4 bytes
  INT4: Each weight = 0.5 bytes ← 8x smaller!

Step 2: Apply LoRA adapters (in higher precision)
  The adapters themselves remain in float16/32

Result:
  7B model needs ~28GB (FP32) → ~4GB (QLoRA)
  70B model needs ~280GB      → ~35GB (fits on 1 consumer GPU!)

This makes fine-tuning frontier models accessible to everyone.
```

---

## 4. OpenAI Fine-Tuning — Complete Tutorial

```python
import os
import json
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Step 1: Prepare Training Data ──

training_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a customer service agent for AcmeCorp. Be concise and professional."},
            {"role": "user", "content": "What is your return policy?"},
            {"role": "assistant", "content": "Returns accepted within 30 days. Item must be unused and in original packaging. Contact returns@acmecorp.com with order number."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a customer service agent for AcmeCorp. Be concise and professional."},
            {"role": "user", "content": "My order hasn't arrived"},
            {"role": "assistant", "content": "I apologise for the delay. Please provide your order number and I'll check the status immediately. For urgent issues, call 1-800-ACME-CORP."}
        ]
    },
    # Add 50-500 more examples for best results
]

# Save as JSONL file (one JSON object per line)
with open("training_data.jsonl", "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

print(f"Created training file with {len(training_examples)} examples")


# ── Step 2: Upload Training File ──

with open("training_data.jsonl", "rb") as f:
    upload_response = client.files.create(
        file=f,
        purpose="fine-tune"
    )

file_id = upload_response.id
print(f"Uploaded file: {file_id}")


# ── Step 3: Create Fine-Tuning Job ──

job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18",   # Model to fine-tune
    hyperparameters={
        "n_epochs": 3,                 # Number of passes through training data
        # More epochs = more specialised but risk of overfitting
        # 1-5 epochs is typical; start with 3
    },
    suffix="customer-service"          # Custom suffix for model name
    # Your model will be named: ft:gpt-4o-mini-...:customer-service
)

job_id = job.id
print(f"Fine-tuning job created: {job_id}")


# ── Step 4: Monitor Training Progress ──

print("Monitoring training...")
while True:
    status = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Status: {status.status}")

    if status.status == "succeeded":
        fine_tuned_model = status.fine_tuned_model
        print(f"\n✓ Training complete! Model: {fine_tuned_model}")
        break

    if status.status in ["failed", "cancelled"]:
        print(f"✗ Training failed: {status.error}")
        break

    # Show training events (loss at each step)
    events = client.fine_tuning.jobs.list_events(job_id, limit=5)
    for event in events.data:
        print(f"  Event: {event.message}")

    time.sleep(60)  # Check every minute


# ── Step 5: Use the Fine-Tuned Model ──

response = client.chat.completions.create(
    model=fine_tuned_model,   # Your custom model!
    messages=[
        {"role": "system", "content": "You are a customer service agent for AcmeCorp."},
        {"role": "user", "content": "How do I track my order?"}
    ]
)
print(f"\nFine-tuned response: {response.choices[0].message.content}")


# ── Cost Calculation ──
# OpenAI charges per token for fine-tuning
# Pricing (approximate): $8 per million training tokens for gpt-4o-mini

def estimate_fine_tuning_cost(
    num_examples: int,
    avg_tokens_per_example: int,
    n_epochs: int
) -> dict:
    total_tokens = num_examples * avg_tokens_per_example * n_epochs
    cost_per_million = 8.00   # $/1M tokens for gpt-4o-mini
    total_cost = (total_tokens / 1_000_000) * cost_per_million

    return {
        "total_training_tokens": total_tokens,
        "estimated_cost": f"${total_cost:.2f}",
        "model": "gpt-4o-mini"
    }

cost = estimate_fine_tuning_cost(
    num_examples=500,
    avg_tokens_per_example=200,
    n_epochs=3
)
print(f"\nEstimated cost: {cost}")
```

---

## 5. HuggingFace + PEFT — LoRA Tutorial

```python
# pip install transformers peft trl bitsandbytes accelerate datasets

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset
import torch


# ── Step 1: Load base model with 4-bit quantization ──

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Configure 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",             # NF4 quantization (best quality)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
    bnb_4bit_use_double_quant=True,        # Double quantize to save more memory
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Needed for batch training

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   # Apply 4-bit loading
    device_map="auto",                 # Automatically place on GPU(s)
)

# ── Step 2: Configure LoRA ──

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # We're fine-tuning a language model
    r=16,                             # LoRA rank: 4-64. Higher = more capacity but more memory
                                      # Start with 16; increase if performance is poor
    lora_alpha=32,                    # Scaling factor. Usually set to 2× rank
    lora_dropout=0.05,               # Dropout regularization (prevents overfitting)
    bias="none",                      # Don't fine-tune bias parameters
    target_modules=[                  # Which layers to add LoRA adapters to
        "q_proj", "v_proj",           # Query and value projections in attention
        "k_proj", "o_proj",           # Key and output projections
        "gate_proj", "up_proj", "down_proj"  # Feed-forward network layers
    ],
)

# Apply LoRA adapters to the model
model = get_peft_model(model, lora_config)

# Show how many parameters we're actually training (should be <1%)
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,072,667,136 || trainable%: 0.5195

# ── Step 3: Prepare Dataset ──

# For Llama 3.1, format with special tokens
def format_instruction(example):
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful assistant.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output']}<|eot_id|>"
    )

# Example dataset (replace with your data)
raw_data = [
    {"instruction": "Explain Python decorators", "output": "Decorators are functions that wrap other functions..."},
    {"instruction": "What is a neural network?", "output": "A neural network is a computational model..."},
    # Add more examples...
]

dataset = Dataset.from_list(raw_data)

# ── Step 4: Configure Training ──

training_args = TrainingArguments(
    output_dir="./llama_lora_output",     # Where to save checkpoints
    num_train_epochs=3,                   # Training passes through data
    per_device_train_batch_size=4,        # Examples per GPU per step
    gradient_accumulation_steps=4,        # Accumulate 4 steps before update
                                          # Effective batch size = 4 × 4 = 16
    warmup_steps=100,                     # Gradually increase LR at start
    learning_rate=2e-4,                   # Learning rate (2e-4 good for LoRA)
    fp16=True,                            # Mixed precision training
    logging_steps=50,                     # Log metrics every 50 steps
    save_strategy="epoch",                # Save checkpoint at each epoch
    report_to="none",                     # "wandb" to use Weights & Biases
    optim="paged_adamw_32bit",            # Memory-efficient optimizer for QLoRA
)

# ── Step 5: Train ──

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_instruction,   # Format examples for the model
    max_seq_length=2048,                   # Max token length per example
)

print("Starting training...")
trainer.train()
print("Training complete!")

# ── Step 6: Save ──

trainer.save_model("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

# ── Optional: Merge LoRA weights into base model ──
# This creates a standalone model that doesn't need PEFT library to load

merged_model = AutoPeftModelForCausalLM.from_pretrained(
    "./my_finetuned_model",
    torch_dtype=torch.float16,
)
merged_model = merged_model.merge_and_unload()  # Merge LoRA into base weights
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
print("Model merged and saved!")
```

---

## 6. RLHF Overview

```
RLHF = Reinforcement Learning from Human Feedback

WHY IT'S NEEDED:
  After SFT, the model follows instructions but may be:
  - Sycophantic (tells users what they want to hear)
  - Unhelpful (technically correct but misses the point)
  - Harmful (provides dangerous information)

THREE PHASES:

Phase 1: SFT (Supervised Fine-Tuning)
  Train on expert demonstrations: instruction → ideal response
  Result: Model that can follow instructions

Phase 2: Reward Model Training
  Show humans pairs of responses to same prompt
  Humans rank which is better
  Train a separate "reward model" to predict human preferences

Phase 3: PPO (Proximal Policy Optimisation)
  Use reward model to score LLM outputs
  Update LLM to maximise reward (while not drifting too far from SFT)
  Result: Model aligned with human preferences

Used by: GPT-4, Claude 3, Gemini — all production LLMs use RLHF or similar
```

---

## 7. DPO (Direct Preference Optimisation)

```python
# DPO is a simpler alternative to RLHF
# Instead of training a separate reward model, train directly on preference pairs

from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# Preference data format: prompt + chosen (better) + rejected (worse) response
preference_data = [
    {
        "prompt": "Explain quantum physics",
        "chosen": "Quantum physics describes the behavior of matter at atomic scales...",
        "rejected": "Quantum physics is really hard and complicated, trust me..."
    },
    {
        "prompt": "Write a poem about rain",
        "chosen": "Silver drops on dusty leaves, the earth drinks deep...",
        "rejected": "Rain falls down from the sky. It is wet. I like rain."
    },
]

dataset = Dataset.from_list(preference_data)

dpo_config = DPOConfig(
    beta=0.1,           # How strongly to deviate from base model
                        # Lower = stay close to base model behavior
                        # Higher = more willing to change
    learning_rate=1e-5,
    num_train_epochs=1,
)

# trainer = DPOTrainer(
#     model=model,
#     ref_model=ref_model,   # Frozen reference model (the base)
#     args=dpo_config,
#     train_dataset=dataset,
# )
# trainer.train()
```

---

## 8. LoRA Hyperparameter Guide

```
RANK (r): Controls capacity of LoRA adapters
──────────────────────────────────────────────
  r=4:   Very lightweight, simple style changes
  r=8:   Light, good for tone/format changes
  r=16:  Standard, good for most tasks (START HERE)
  r=32:  More capacity, complex domain adaptation
  r=64+: High capacity, large training datasets

ALPHA: Scaling factor for LoRA updates
──────────────────────────────────────
  Rule: alpha = 2 × rank (e.g., r=16, alpha=32)
  Higher alpha = larger updates to base model
  If outputs are too generic: try higher alpha
  If outputs are too strange: try lower alpha

LEARNING RATE:
────────────────────────────────────────────────
  For LoRA: 1e-4 to 5e-4 (higher than full fine-tuning)
  Start: 2e-4
  Adjust: if loss doesn't decrease, try higher
           if loss oscillates, try lower

EPOCHS:
───────────────────────────────────────────────
  1 epoch:  If you have 1000+ examples
  3 epochs: Standard starting point
  5 epochs: If you have <100 examples
  Watch validation loss: stop if it increases (overfitting)
```

---

## Key Points for Exam Prep

```
FINE-TUNING CHEAT SHEET:
  - Fine-tune for: style, format, domain vocab, reduced prompts
  - DON'T fine-tune for: knowledge (use RAG), one-time tasks
  - LoRA: freeze base, add tiny A×B matrices, only train those
  - QLoRA: 4-bit quantization + LoRA = runs on consumer GPU
  - OpenAI format: JSONL with {"messages": [...]} per line
  - Minimum: 50 examples; ideal: 200-500 quality examples
  - n_epochs: 3 is a good starting point
  - RLHF: SFT → Reward Model → PPO → aligned model
  - DPO: simpler alternative to RLHF using preference pairs
  - Always evaluate before/after: compare on held-out test set
```

## Practice Questions

1. What is the difference between fine-tuning and RAG?
2. What does LoRA do and why is it more efficient than full fine-tuning?
3. What is QLoRA and why does it make 70B model fine-tuning accessible?
4. What format does OpenAI require for fine-tuning data?
5. What are the three phases of RLHF?
6. What is DPO and how does it differ from RLHF?
7. What minimum number of examples do you need for OpenAI fine-tuning?
8. What is catastrophic forgetting and how does LoRA help avoid it?
9. What does the rank (r) parameter in LoRA control?
10. How many trainable parameters does a rank-16 LoRA have vs full fine-tuning?
11. When would you increase the number of training epochs?
12. What is the difference between SFT and RLHF in training?
13. How do you evaluate if fine-tuning improved your model?
14. What is the OpenAI fine-tuning file_id used for?
15. What is the `merge_and_unload()` step and why would you use it?

---
*Next: [17 — LLMOps](../17-llmops/README.md)*
