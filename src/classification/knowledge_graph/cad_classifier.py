import json
import pandas as pd
import urllib.parse
from unittest import result
from rdflib import Graph, URIRef, Literal, Namespace
from ollama import chat, ChatResponse
from pathlib import Path
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent
input_path = root_dir / "ethos_dataset_withContext_tokenized.json"
output_file_path = root_dir / "classified_ethos_dataset_withContext_tokenized.json"
input_kg_turtle_path = (
    root_dir.parent.parent
    / "knowledge_graph_hate_speech"
    / "knowledge_graph_turtle.ttl"
)

# Define your namespace
ns = Namespace("http://hate_speech_detection.org/")

tqdm.pandas()
loaded_g = Graph()
loaded_g.parse(input_kg_turtle_path, format="turtle")


def get_dataset():
    data = pd.read_json(input_path, orient="records")
    return data


def shorten_uri(uri):
    """Removes namespace and decodes URL-encoded characters."""
    # Remove namespace if it starts with it
    if uri.startswith(ns):
        uri = uri[len(ns) :]

    # Decode URL-encoded characters (e.g., %20 -> space)
    return urllib.parse.unquote(uri)


def query_kg(token):
    # Ensure the token is lowercased for case-insensitive matching
    token = token.lower()

    # Properly format the token inside the SPARQL query
    query = f"""
        SELECT ?s ?p ?o WHERE {{
            ?s ?p ?o .
            FILTER (
                CONTAINS(LCASE(STR(?s)), "{token}") ||
                CONTAINS(LCASE(STR(?p)), "{token}") ||
                CONTAINS(LCASE(STR(?o)), "{token}")
            )
        }}
    """

    # Execute the query on the knowledge graph
    return loaded_g.query(query)


def retreive_connections(tokens, commonTokens):
    foundPath = []
    for token in commonTokens:
        result = query_kg(token)
        for s, p, o in result:
            s = shorten_uri(s)
            p = shorten_uri(p)
            o = shorten_uri(o)

            if s in commonTokens:
                if p in tokens or o in tokens:
                    foundPath.append(s + " " + p + " " + o + " ")

            elif o in commonTokens:
                if p in tokens or s in tokens:
                    foundPath.append(s + " " + p + " " + o + " ")
    return foundPath


# Function to classify a comment
def classify_comment(comment, model, tokens, common_tokens):
    common_pattern = retreive_connections(tokens, common_tokens)

    # Example Task Instruction
    examples = """
    The following comment needs to be classified.

    Task: Determine whether the comment contains hate speech against people with disabilities.

    Response Format: Reply only with **'yes'** or **'no'**. Do not provide explanations or generate any additional text.
    """

    # System Context for Model
    context = """
    You are an expert in hate speech detection against people with disabilities.
    
    Classification Guidelines:
    - Respond only with "yes" or "no". Do not provide explanations, justifications, or any additional text.
    - Only comments against people with disabilities must be classified as "yes".
    - Hateful comments not directed against people with disabilities must be classified as "no".
    """

    # Construct the Prompt
    if len(common_pattern) > 0:
        prompt = f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.\nNote: This comment contains words that commonly appear in hate speech patterns as follows: {common_pattern}\n\nComment: {comment}\nResponse:"
    else:
        prompt = f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.\nComment: {comment}"

    # Generate Response
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    # Extract and Return Classification Result
    classification = response["message"]["content"].strip().lower()
    print(classification)
    return classification


df = get_dataset()

# print("Starting classification with llama model...")
df["result_llama"] = df.progress_apply(
    lambda row: classify_comment(
        row["comment"], "llama3.1:8b", row["tokenized_text"], row["common_tokens"]
    ),
    axis=1,
)


print("Saving the result......")
df.to_json(output_file_path, orient="records", force_ascii=False)
