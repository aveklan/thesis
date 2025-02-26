import pandas as pd
import spacy
import networkx as nx
import numpy as np
import re
from pathlib import Path
from pyvis.network import Network
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from tqdm import tqdm
from collections import Counter


nlp = spacy.load("en_core_web_md")

root_dir = Path(__file__).resolve().parent
input_path = root_dir / "hate_speech_KG_dataset_only_comments.json"
nodes_output_path = root_dir / "KG_nodes.csv"
edges_output_path = root_dir / "KG_edges.csv"
grap_clustered_output_path = root_dir / "GK_graph_clustered.html"
grap_unclustered_output_path = root_dir / "GK_graph_unclustered.html"
search_grap_output_path = root_dir / "GK_search_graph.html"
comments_with_common_words_output_path = (
    root_dir / "hate_speech_KG_dataset_comments_with_common_words.json"
)

tqdm.pandas()


def remove_mentions(comment):
    text = re.sub(r"@\w+", "you", comment.lower())
    return text


def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]


# def filter_frequent_words(comments, min_occurrences=3):
#     all_words = [word for comment in comments for word in preprocess_text(comment)]
#     word_counts = Counter(all_words)
#     return [word for word, count in word_counts.items() if count >= min_occurrences]


def filter_frequent_words(all_tokens, min_occurrences=3):
    """
    Filters tokens that appear at least `min_occurrences` times.

    :param all_tokens: A flattened list of all tokens from comments.
    :param min_occurrences: Minimum number of times a token must appear to be included.
    :return: A list of high-frequency tokens.
    """
    word_counts = Counter(all_tokens)  # Count token occurrences
    return [word for word, count in word_counts.items() if count >= min_occurrences]


def cluster_similar_words(unique_tokens):
    # Convert words into word embeddings
    word_vectors = np.array([nlp(token).vector for token in unique_tokens])

    # Cluster similar words using KMeans
    num_clusters = int(len(unique_tokens) * 0.1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(word_vectors)

    # Create a mapping of cluster -> words
    cluster_dict = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        cluster_dict[cluster_id].append(unique_tokens[idx])
    return cluster_dict


def build_cooccurrence_graph(comments, cluster_dict, window_size=3):
    G = nx.Graph()

    for comment in comments:
        tokens = preprocess_text(comment)

        # Case 1: If cluster_dict is a dictionary (word clustering case)
        if isinstance(cluster_dict, dict) and all(
            isinstance(words, list) for words in cluster_dict.values()
        ):
            reduced_tokens = set()
            for token in tokens:
                for cluster_id, words in cluster_dict.items():
                    if token in words:
                        reduced_tokens.add(
                            words[0]
                        )  # Use the first word as the cluster representative
            reduced_tokens = list(reduced_tokens)  # Convert set to list

        # Case 2: If cluster_dict is a list (single tokens case)
        elif isinstance(cluster_dict, list):
            reduced_tokens = cluster_dict  # Directly use the token list

        else:
            raise TypeError(
                "Invalid format for cluster_dict. It must be a dictionary of lists or a list of tokens."
            )

        # Connect words in the same comment
        for i in range(len(reduced_tokens)):
            for j in range(i + 1, min(i + window_size, len(reduced_tokens))):
                if G.has_edge(reduced_tokens[i], reduced_tokens[j]):
                    G[reduced_tokens[i]][reduced_tokens[j]]["weight"] += 1
                else:
                    G.add_edge(reduced_tokens[i], reduced_tokens[j], weight=1)

    return G


def filter_graph(G, edge_weight_threshold=5, min_node_degree=3):
    # Remove edges with weight < edge_weight_threshold
    filtered_G = nx.Graph()
    filtered_G.add_nodes_from(G.nodes(data=True))  # Keep all nodes initially

    for u, v, data in G.edges(data=True):
        if data["weight"] >= edge_weight_threshold:
            filtered_G.add_edge(u, v, weight=data["weight"])

    # Remove nodes with degree ≤ min_node_degree
    nodes_to_remove = [
        node
        for node, degree in dict(filtered_G.degree()).items()
        if degree <= min_node_degree
    ]
    filtered_G.remove_nodes_from(nodes_to_remove)

    return filtered_G


def find_common_words_from_comment(comment, high_frequency_set):
    tokens = []
    tokens.extend(preprocess_text(comment))
    words = set(tokens)
    high_frequency_set = set(high_frequency_set)
    return list(words & high_frequency_set)


def visualize_graph_interactive(G):
    net = Network(
        notebook=True,
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
    )

    for node in G.nodes():
        net.add_node(node, label=node)

    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, value=data["weight"])

    net.set_options(
        """
    var options = {
      "physics": {
        "enabled": false
      }
    }
    """
    )

    net.show(str(grap_unclustered_output_path))


df = pd.read_json(input_path, orient="records")

print("Removing mentions from comments......")
comments = df[0]
preprocessed_comments = comments.astype(str).progress_apply(remove_mentions)

print("Tokenizing the comments......")
# Apply `preprocess_text()` to each comment and store in a new column
df["tokenized_text"] = comments.astype(str).progress_apply(preprocess_text)
# Get a flattened list of all unique tokens
all_tokens = [
    token for tokens in df["tokenized_text"] for token in tokens
]  # Flatten list
unique_tokens = list(
    set([token for tokens in df["tokenized_text"] for token in tokens])
)

# Get tokens that appears more than n times in the comments
print("Extracting tokens with high frequency......")
high_fequency_tokens = filter_frequent_words(all_tokens)  # Apply frequency filter
unique_tokens_high_frequency = list(set(high_fequency_tokens))

print("Creating word clusters......")
word_clusters = cluster_similar_words(unique_tokens)

print("Creating word graph......")
graph = build_cooccurrence_graph(comments, unique_tokens_high_frequency)
filtered_graph = filter_graph(graph)

print(
    "Initial number of tokens: ",
    len(all_tokens),
    "\nUnunique Tokens: ",
    len(unique_tokens),
    "\nUnunique Tokens with frequency higher than 3: ",
    len(unique_tokens_high_frequency),
    "\nNumber of world clusters: ",
    len(word_clusters),
    "\nOriginal Graph Nodes: ",
    graph.number_of_nodes(),
    "\nFiltered Graph nodes: ",
    filtered_graph.number_of_nodes(),
)

# Create a new column in the df which only contains the common words used that comment
print("Finding common words in ech comment...")
df["originalComment"] = comments
df["preprocessedComments"] = preprocessed_comments
df["commonEdges"] = comments.astype(str).progress_apply(
    lambda x: find_common_words_from_comment(x, unique_tokens_high_frequency)
)

df.drop(df.columns[0], axis=1).to_json(
    comments_with_common_words_output_path, orient="records", force_ascii=False
)


visualize_graph_interactive(filtered_graph)
# Example: Display connections for the word "hate"
# save_graph(graph)
