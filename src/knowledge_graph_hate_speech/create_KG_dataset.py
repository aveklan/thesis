import pandas as pd
from pathlib import Path

# Define the root directory
root_dir = Path(__file__).resolve().parent.parent
input_path = (
    root_dir
    / "dataset"
    / "cleaned_json_datasets"
    / "measuring_hate_speech_dataset_withContext_cleaned.json"
)
output_path = root_dir / "knowledge_graph_hate_speech" / "hate_speech_KG_dataset.csv"
output_json_path = (
    root_dir
    / "knowledge_graph_hate_speech"
    / "hate_speech_KG_dataset_only_comments.json"
)

df = pd.read_json(input_path, orient="records")

# Display the first few rows to confirm
print(df.head(), "\nFile length: ", len(df))

df = df[df["hate_speech_score"] >= 0.5]
df = df[df["target_disability"] == True]

# Display the first few rows to confirm
print(df.head(), "\nFile length only for hate speech: ", len(df))

df.to_csv(output_path, index=False)

df = df["text"]
print(df.head(), "\nFile length only for hate speech: ", len(df))

# Remove empty or duplicate comments
df = df.astype(str).str.lower()
comments_non_empty = df.dropna()
comments_without_duplicates = comments_non_empty.drop_duplicates()
print("Comments size after removing empty comments: ", comments_non_empty.size)
print("Comments size after removing duplicates: ", comments_without_duplicates.size)

# Save cleaned dataset
comments_without_duplicates.to_json(
    output_json_path, orient="records", force_ascii=False
)
