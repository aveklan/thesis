from html import entities
import pandas as pd
import re
import spacy
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from textblob import TextBlob
from openie import StanfordOpenIE
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

nlp = spacy.load("en_core_web_md")  # Load a pre-trained spaCy model

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

root_dir = Path(__file__).resolve().parent
input_path = root_dir / "hate_speech_KG_dataset_only_comments.json"
output_json_path_relationships = root_dir / "extracted_relationships.json"
output_json_path_relationships_babelscapeGPT = (
    root_dir / "extracted_relationships_babelscapeGPT.json"
)
output_json_path_relationships_babelscapeGIT = (
    root_dir / "extracted_relationships_babelscapeGIT.json"
)
output_json_path_relationships_babelscapeGITBatch = (
    root_dir / "extracted_relationships_babelscapeGITBatch.json"
)

abbreviation_dict = {
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "idk": "I don't know",
    "imho": "in my humble opinion",
    "brb": "be right back",
    "lol": "laugh out loud",
    "tbh": "to be honest",
    "nvm": "never mind",
    "omg": "oh my god",
    "thx": "thanks",
    "ty": "thank you",
    "pls": "please",
    "msg": "message",
    "gr8": "great",
    "im": "I am",
    "gonna": "going to",
    "youve": "you have",
    "youre": "you are",
}
pattern = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in abbreviation_dict.keys()) + r")\b",
    re.IGNORECASE,
)

tqdm.pandas()

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

# Load the REBEL model
triplet_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large",
    tokenizer="Babelscape/rebel-large",
    device=0,  # Uses GPU if available (-1 for CPU)
)


##### 1. Load dataset #####
def load_dataset(dataset_path):
    """
    Loads the given dataset by using pandas library.
    Accepts dataset in json format.
    """
    df = pd.read_json(dataset_path, orient="records")
    comments = df[0]
    return comments


##### 2. Data Preprocessing #####
def preprocess_text(text):
    """
    Clean the data: Remove duplicates, handle missing values, and standardize formats.
    """
    print("Preprocessing test...")
    print(
        "Phase 1: Remove duplicates, handle missing values, and standardize formats..."
    )

    # Convert comments to lowecase value
    lowercase_comments = text.astype(str).str.lower()

    # Remove empty or duplicate comments
    comments_non_empty = lowercase_comments.dropna()
    comments_without_duplicates = comments_non_empty.drop_duplicates()
    print("Comments size after removing empty comments: ", comments_non_empty.size)
    print("Comments size after removing duplicates: ", comments_without_duplicates.size)

    # Remove special characters
    print("Remove special characters")
    comments_without_special_characters = comments_without_duplicates.progress_apply(
        lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x, flags=re.MULTILINE)
    )  # Remove URLs
    # comments_without_special_characters = comments_without_special_characters.apply(
    #     lambda x: re.sub(r"\@\w+|\#\w+", "", x)
    # )  # Remove mentions and hashtags
    comments_without_special_characters = (
        comments_without_special_characters.progress_apply(
            lambda x: re.sub(r"[^\w\s]", "", x)
        )
    )  # Remove punctuation

    # Use a dictionary to replace common abbreviations (e.g., "u" → "you", "btw" → "by the way").
    print("Replace common abbreviations")
    comments_without_abbreviations = comments_without_special_characters.astype(
        str
    ).progress_apply(
        lambda x: pattern.sub(lambda match: abbreviation_dict[match.group().lower()], x)
    )

    # Correct grammar errors
    print("Correct grammar errors")
    comments_without_grammar_errors = comments_without_abbreviations.progress_apply(
        lambda x: str(TextBlob(x).correct())
    )

    print("Done prepocessing text...")
    return comments_without_grammar_errors


##### 3. Entity Extraction #####
def entity_extraction(comments):
    """
    Performs extity extraction given a set of comments.
    """
    print("Phase 3, entity extraction...")
    # Remove stopwords
    comments_without_stopwords = comments.astype(str).progress_apply(
        lambda x: " ".join(
            [word for word in x.split() if word.lower() not in stop_words]
        )
    )

    # extracted_entities = comments_without_stopwords.astype(str).apply(
    #     lambda text: [(ent.text, ent.label_) for ent in nlp(text).ents]
    # )

    # Performing NER
    extracted_entities = comments_without_stopwords.astype(str).progress_apply(nlp)

    print("Done entity extraction...")
    return extracted_entities


##### 3. Relationships Extraction #####
def extract_relationships_automated(comments):
    print("Phase 4, relationships extraction...")

    with StanfordOpenIE(timeout=30000) as client:
        extracted_relationships = comments.astype(str).progress_apply(
            lambda x: client.annotate(x)
        )
    non_empty_extracted_relationships = extracted_relationships[
        extracted_relationships.str.len() > 0
    ]
    print(non_empty_extracted_relationships.head(10))

    print("Done relationships extraction...")
    return non_empty_extracted_relationships


def extract_relationship_from_rebel(comments):
    print("Extracting relationships with rebel model...")
    extracted_relations = comments.astype(str).progress_apply(
        lambda x: tokenizer.decode(
            model.generate(**tokenizer(x, return_tensors="pt"))[0],
            skip_special_tokens=True,
        )
    )

    return extracted_relations


def extract_relationships_from_rebel_GIT(comments):
    print("Extracting relationships with REBEL model with GIT script...")

    extracted_relations = comments.astype(str).progress_apply(
        lambda x: extract_triplets(
            triplet_extractor.tokenizer.batch_decode(
                [
                    triplet_extractor(x, return_tensors=True, return_text=False)[0][
                        "generated_token_ids"
                    ]
                ]
            )[0]
        )
    )
    print("Done extracting relationships GIT.", extracted_relations.head())
    return extracted_relations


def extract_relationships_from_rebel_GIT_batch(comments, batch_size=16):
    print("Extracting relationships with REBEL model using batch processing...")

    # Convert comments to strings and process them in batches
    extracted_relations = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i : i + batch_size].tolist()
        outputs = triplet_extractor(batch, return_tensors=True, return_text=False)

        decoded_texts = triplet_extractor.tokenizer.batch_decode(
            [out["generated_token_ids"] for out in outputs]
        )
        extracted_relations.extend([extract_triplets(text) for text in decoded_texts])

    return extracted_relations


# Function to parse the generated text and extract triplets
def extract_triplets(text):
    triplets = []
    relation, subject, object_ = "", "", ""
    text = text.strip()
    current = "x"

    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation:
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            subject = ""
            relation = ""
        elif token == "<subj>":
            current = "s"
            if relation:
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject and relation and object_:
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )

    return triplets


dataset_elements = load_dataset(input_path)
print("Dataset loaded correctly, element loaded: ", dataset_elements.size)

preprocessed_comments = preprocess_text(dataset_elements)
print(
    "Preprocess phase performed correctly, remaining elements: ",
    preprocessed_comments.size,
)

extracted_entities = entity_extraction(preprocessed_comments)
print(
    "Entity extraction phase performed correctly, remaining elements: ",
    extracted_entities.size,
)

extracted_relationships = extract_relationships_automated(preprocessed_comments)
# extracted_relationships_rebel = extract_relationship_from_rebel(preprocessed_comments)
extracted_relationships_rebel_GIT = extract_relationships_from_rebel_GIT(
    preprocessed_comments
)
extracted_relationships_rebel_GIT_batch = extract_relationships_from_rebel_GIT_batch(
    preprocessed_comments
)

print(
    "Relationships extraction phase performed correctly, remaining elements: ",
    extracted_relationships_rebel_GIT.head(),
)
extracted_relationships.to_json(
    output_json_path_relationships, orient="records", indent=4, force_ascii=False
)
# extracted_relationships_rebel.to_json(
#     output_json_path_relationships_babelscapeGPT,
#     orient="records",
#     indent=4,
#     force_ascii=False,
# )

try:
    extracted_relationships_rebel_GIT.to_json(
        output_json_path_relationships_babelscapeGIT,
        orient="records",
        indent=4,
        force_ascii=False,
    )
    print(
        f"Successfully saved extracted relationships to {output_json_path_relationships_babelscapeGIT}"
    )
except Exception as e:
    print(f"Error saving extracted_relationships_rebel_GIT: {e}")

try:
    # Convert the list to a pandas DataFrame
    df_extracted_relationships = pd.DataFrame(
        extracted_relationships_rebel_GIT_batch, columns=["relationships"]
    )

    # Save the DataFrame to JSON
    df_extracted_relationships.to_json(
        output_json_path_relationships_babelscapeGITBatch,
        orient="records",
        indent=4,
        force_ascii=False,
    )
    print(
        f"Successfully saved batch relationships to {output_json_path_relationships_babelscapeGITBatch}"
    )
except Exception as e:
    print(f"Error saving extracted_relationships_rebel_GIT_batch: {e}")

print(f"Extracted relationships saved to {output_json_path_relationships}")
