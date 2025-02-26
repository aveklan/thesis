import pandas as pd
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import json
import re

tqdm.pandas()


keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]
models = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "deepseek/deepseek-chat:free",
    "qwen/qwen-vl-plus:free",
]


root_dir = Path(__file__).resolve().parent
input_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships_new.json"
)
output_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships_new.json"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=keys[16],
)


def load_dataset(dataset_path):
    """
    Loads the given dataset by using pandas library.
    Accepts dataset in json format.
    """
    df = pd.read_json(dataset_path, orient="records")
    return df


def generate_relationships(comment, keywords):
    completion = client.chat.completions.create(
        model=models[2],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in extracting RDF triples (subject-predicate-object) from hateful online comments. "
                    "Your goal is to support hate speech research by identifying relationships between hateful terms.\n\n"
                    "Guidelines:\n"
                    "- Each triple must include at least one of the provided keywords as either the predicate or the object.\n"
                    "- The predicate is preferred be a verb.\n"
                    "- Ensure extracted triples maintain the original meaning and context of the comment.\n"
                    "- Subject and object cannot be the same word in the same RDF triple"
                    "- Return the extracted RDF triples in valid JSON format, structured as a list of { 'subject': '', 'predicate': '', 'object': '' } dictionaries.\n"
                    "- If no valid triples can be extracted from a comment, return an empty list []."
                ),
            },
            {
                "role": "user",
                "content": f"""Task: Extract RDF triples (subject-predicate-object) from the given comment. Each triple must contain at least one of the provided keywords as either the predicate or the object. Return the triples in JSON format.

    Input:
    - Comment: "{comment}"
    - Keywords: "{keywords}"

    """,
            },
        ],
        max_tokens=10000,
    )
    return completion.choices[0].message.content


# Load dataframe
df = load_dataset(input_path)
# Ensure 'extracted_relationships' column exists in the DataFrame
if "extracted_relationships" not in df.columns:
    df["extracted_relationships"] = None  # Initialize with None

# Iterate over each row in the DataFrame with progress tracking
for index, row in tqdm(df.iterrows(), total=len(df)):
    # If the column does not exist, create it for this row
    extracted_relationships = row.get("extracted_relationships", None)
    if isinstance(extracted_relationships, list) and len(extracted_relationships) > 0:
        print("Skipping: ", index)
        continue  # Skip if relationships already exist
    else:
        # Call the API and get the triples
        triples = generate_relationships(
            row["preprocessedComments"], row["commonEdges"]
        )
        print(triples)

        # Try parsing JSON safely
        try:
            # Remove Markdown-style triple backticks and "json" label
            cleaned_triples = re.sub(r"```json|```", "", triples).strip()
            parsed_data = json.loads(cleaned_triples)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed at index {index}: {e}")
            print(f"Problematic JSON: {cleaned_triples}")
            parsed_data = None  # Use an empty list if JSON parsing fails

        df.at[index, "extracted_relationships"] = parsed_data
        # Update the current row with the extracted relationships
        df.to_json(output_path, orient="records", force_ascii=False)

print("DataFrame updated and saved incrementally to output.json")
