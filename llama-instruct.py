
from huggingface_hub import login
import transformers
import torch

login(token="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")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=50,
)
print(outputs[0]["generated_text"][-1])
