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
]
models = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "deepseek/deepseek-chat:free",
]


root_dir = Path(__file__).resolve().parent
input_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships.json"
)
output_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships.json"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=keys[3],
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
