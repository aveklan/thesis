import json
import re
import pandas as pd
from pathlib import Path

# Define the root directory
root_dir = Path(__file__).resolve().parent
input_path = root_dir / "raw_data" / "measuring_hate_speech.csv"
output_path = (
    root_dir
    / "cleaned_json_datasets"
    / "measuring_hate_speech_dataset_withContext_cleaned.json"
)

df = pd.read_csv(input_path)

print(df.head(), "\nFile length: ", len(df))

# Select only the desired columns
df = df[["comment_id", "hate_speech_score", "text", "target_disability"]]

# Display the first few rows to confirm
print(df.head(), "\nFile length: ", len(df))


# Save cleaned dataset
df.to_json(output_path, orient="records", lines=True)
