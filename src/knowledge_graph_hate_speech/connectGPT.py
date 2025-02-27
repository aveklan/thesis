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
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships_third_version.json"
)
output_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships_third_version.json"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=keys[6],
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
        model=models[3],
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
                    "- Do not provide any additional explanation, just return the RDF triples in json format."
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
        triples = generate_relationships(row["originalComment"], row["commonEdges"])
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
