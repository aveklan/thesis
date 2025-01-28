from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from evaluate import load
from pathlib import Path
import torch

# Paths
root_dir = Path(__file__).resolve().parent.parent
model_id = "meta-llama/Meta-Llama-3.1-8B"
trained_model_id = (
    root_dir
    / "fine_tuning"
    / "Llama-3.1-8_V4-one_epoch_new_settings-sft-lora-fine_tuned"
)
output_dir = trained_model_id

# Dataset
dataset = load_dataset(
    "json",
    data_files=str(
        root_dir / "dataset" / "fine_tuning_datasets" / "combined_training_dataset.json"
    ),
)
testing_dataset = load_dataset(
    "json",
    data_files=str(
        root_dir / "dataset" / "fine_tuning_datasets" / "combined_testing_dataset.json"
    ),
)

train_dataset = dataset["train"]
eval_dataset = testing_dataset["train"]

# Hugging Face login
login(token="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")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# Quantization Config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)


# Model Loading
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    num_labels=2,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "10.0GB"},  # Limit GPU memory usage to 11GB
    revision="main",
)

# Apply LoRA Config
peft_config = LoraConfig(
    r=32,  # Increase from 16 for more capacity
    lora_alpha=32,  # Keep alpha = 2 * r
    lora_dropout=0.05,  # Reduce dropout
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, peft_config)

# Training Arguments
training_args = TrainingArguments(
    fp16=False,  # Use bf16 on supported GPUs
    bf16=False,
    do_eval=False,
    evaluation_strategy="no",  # No evaluation
    eval_steps=200,  # Increased to reduce memory pressure
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # Add warmup period
    logging_steps=5,
    logging_strategy="steps",
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1,
    auto_find_batch_size=True,
    push_to_hub=False,
    hub_model_id=trained_model_id,
    report_to="none",
    save_strategy="steps",
    save_steps=200,  # Increased to reduce memory pressure
    max_grad_norm=0.3,  # Added to help stabilize training
    max_steps=-1,  # -1 means use num_train_epochs
    seed=42,
)


def compute_metrics(pred):
    metric_f1 = load("f1")
    metric_accuracy = load("accuracy")
    logits, labels = pred
    predictions = logits.argmax(axis=-1)
    f1 = metric_f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    return {"f1": f1["f1"], "accuracy": accuracy["accuracy"]}


# Trainer
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # max_seq_length=512,  # Add sequence length limit
    compute_metrics=compute_metrics,  # Add custom metrics computation
)

# Train
torch.cuda.empty_cache()
train_result = trainer.train()

# Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Evaluate and display metrics
# metrics = trainer.evaluate()
# print("Evaluation Metrics:", metrics)
