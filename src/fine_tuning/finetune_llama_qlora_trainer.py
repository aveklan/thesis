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
from pathlib import Path
import torch

# Paths
root_dir = Path(__file__).resolve().parent.parent
model_id = "meta-llama/Meta-Llama-3.1-8B"
trained_model_id = root_dir / "fine_tuning" / "Llama-3.1-8B-sft-lora-fine_tuned"
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
## For 4 bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# For 8 bit quantization
# quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0)

# Model Loading
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    num_labels=2,
    quantization_config=quantization_config,
    device_map="auto",
    revision="main",
)

# Apply LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, peft_config)

# Training Arguments
training_args = TrainingArguments(
    fp16=False,  # Use bf16 on supported GPUs
    bf16=False,
    do_eval=True,
    evaluation_strategy="no",  # No evaluation
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    push_to_hub=False,
    hub_model_id=trained_model_id,
    report_to="none",
    save_strategy="no",
    seed=42,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
torch.cuda.empty_cache()
train_result = trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
