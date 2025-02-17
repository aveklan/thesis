import pandas as pd
import re
import spacy
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from textblob import TextBlob


nlp = spacy.load("en_core_web_md")  # Load a pre-trained spaCy model

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

root_dir = Path(__file__).resolve().parent
input_path = root_dir / "hate_speech_KG_dataset_only_comments.json"

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
}
pattern = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in abbreviation_dict.keys()) + r")\b",
    re.IGNORECASE,
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
    comments_without_special_characters = comments_without_duplicates.apply(
        lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x, flags=re.MULTILINE)
    )  # Remove URLs
    # comments_without_special_characters = comments_without_special_characters.apply(
    #     lambda x: re.sub(r"\@\w+|\#\w+", "", x)
    # )  # Remove mentions and hashtags
    comments_without_special_characters = comments_without_special_characters.apply(
        lambda x: re.sub(r"[^\w\s]", "", x)
    )  # Remove punctuation

    # Use a dictionary to replace common abbreviations (e.g., "u" → "you", "btw" → "by the way").
    comments_without_abbreviations = comments_without_special_characters.astype(
        str
    ).apply(
        lambda x: pattern.sub(lambda match: abbreviation_dict[match.group().lower()], x)
    )

    print("Done prepocessing text...")
    return comments_without_abbreviations


##### 3. Entity Extraction #####
def entity_extraction(comments):
    """
    Performs extity extraction given a set of comments.
    """
    print("Phase 3, entity extraction...")
    # Remove stopwords
    comments_without_stopwords = comments.astype(str).apply(
        lambda x: " ".join(
            [word for word in x.split() if word.lower() not in stop_words]
        )
    )

    # extracted_entities = comments_without_stopwords.astype(str).apply(
    #     lambda text: [(ent.text, ent.label_) for ent in nlp(text).ents]
    # )

    extracted_entities = comments_without_stopwords.astype(str).apply(nlp)

    print(extracted_entities)
    print("Done entity extraction...")
    return extracted_entities


dataset_elements = load_dataset(input_path)
print("Dataset loaded correctly, element loaded: ", dataset_elements.size)

preprocessed_comments = preprocess_text(dataset_elements)
print(
    "Preprocess phase performed correctly, remaining elements: ", dataset_elements.size
)

entity_extraction(preprocessed_comments)
