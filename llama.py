from huggingface_hub import login
import transformers
import torch

login(token="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")

model_id = "meta-llama/Llama-3.1-8B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)

response = pipeline("What is hate speech? I want a precise definition and some general considerations about it", max_length=100)
print(response)