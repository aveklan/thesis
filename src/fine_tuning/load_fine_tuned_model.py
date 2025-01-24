from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import torch

# Load the fine-tuned model and tokenizer
model_path = "Llama-3.1-8B-sft-lora-fine_tuned" # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)

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

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically map layers to GPU
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Function to classify a comment
def classify_comment(comment, model):
    # Tokenize the input comment
    inputs = tokenizer(
        comment,
        return_tensors="pt",
        truncation=True,
        padding="longest",  # Ensures proper padding
        max_length=512,
    )
    
    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Convert predicted class to label (assuming binary classification: 0 = No, 1 = Yes)
    label_map = {0: "No", 1: "Yes"}
    return label_map[predicted_class]


# Example usage
if __name__ == "__main__":
    test_comment = "You stupid disabled"
    result = classify_comment(test_comment, model)
    print(f"Comment: {test_comment}")
    print(f"Classification: {result}")
