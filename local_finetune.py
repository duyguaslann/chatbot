"""
Llama-3.2-3B uzerinde QLoRA ile fine-tuning.
MLFlow entegrasyonu ile parametre ve metrik loglama.
Kullanim: python local_finetune.py
GPU yoksa CPU'da calisir (yavas olur, uyari verir).
"""
import json
import math
import os

import mlflow
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

load_dotenv()

MODEL_ID    = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-3B")
JSONL_PATH  = os.getenv("TRAINING_DATA", "training_data.jsonl")
OUTPUT_DIR  = "./fine_tuned_model"
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

HAS_GPU = torch.cuda.is_available()
if not HAS_GPU:
    print("UYARI: GPU bulunamadi. CPU uzerinde egitim cok yavas olabilir.")

# ── Veri ──────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

records = load_jsonl(JSONL_PATH)
print(f"Egitim ornegi: {len(records)} satir")

dataset = Dataset.from_list(records)

# ── Tokenizer ─────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    text = f"### Human: {example['prompt']}\n### Assistant: {example['completion']}"
    out  = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tokenize, remove_columns=["prompt", "completion"])

# train / eval split
split     = tokenized.train_test_split(test_size=0.1, seed=42)
train_ds  = split["train"]
eval_ds   = split["test"]

# ── Model ─────────────────────────────────────────────────────────────────────

if HAS_GPU:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

# ── LoRA ──────────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training args ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=HAS_GPU,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

# ── MLFlow callback ───────────────────────────────────────────────────────────

class MLFlowEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        for entry in reversed(state.log_history):
            if "loss" in entry:
                mlflow.log_metric("train_loss", entry["loss"], step=int(state.epoch))
                break

# ── Egitim ────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("cma-llm-finetune")

with mlflow.start_run():
    mlflow.log_params({
        "model_id":       MODEL_ID,
        "lora_r":         16,
        "lora_alpha":     32,
        "lora_dropout":   0.05,
        "target_modules": "q_proj,v_proj",
        "epochs":         3,
        "learning_rate":  2e-4,
        "batch_size":     4,
        "device":         "cuda" if HAS_GPU else "cpu",
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[MLFlowEpochCallback()],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # Perplexity (eval loss'tan)
    eval_result = trainer.evaluate()
    eval_loss   = eval_result.get("eval_loss", float("nan"))
    perplexity  = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")

    print(f"\nEval loss  : {eval_loss:.4f}")
    print(f"Perplexity : {perplexity:.2f}")

    mlflow.log_metric("eval_loss",  eval_loss)
    mlflow.log_metric("perplexity", perplexity)
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="adapter")

print(f"\nEgitim tamamlandi. Model: {OUTPUT_DIR}")
