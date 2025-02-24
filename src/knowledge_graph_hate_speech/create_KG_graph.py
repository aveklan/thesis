from enum import unique
from networkx import number_of_nodes
import pandas as pd
import spacy
import csv
import networkx as nx
import numpy as np
from pathlib import Path
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict, Counter


nlp = spacy.load("en_core_web_md")

root_dir = Path(__file__).resolve().parent
input_path = root_dir / "hate_speech_KG_dataset_only_comments.json"
nodes_output_path = root_dir / "KG_nodes.csv"
edges_output_path = root_dir / "KG_edges.csv"
grap_clustered_output_path = root_dir / "GK_graph_clustered.html"
grap_unclustered_output_path = root_dir / "GK_graph_unclustered.html"
search_grap_output_path = root_dir / "GK_search_graph.html"


def preprocess_text(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]


def filter_frequent_words(comments, min_occurrences=3):
    all_words = [word for comment in comments for word in preprocess_text(comment)]
    word_counts = Counter(all_words)
    return [word for word, count in word_counts.items() if count >= min_occurrences]


def cluster_similar_words(unique_tokens):
    # Convert words into word embeddings
    word_vectors = np.array([nlp(token).vector for token in unique_tokens])

    # Cluster similar words using KMeans
    num_clusters = int(len(unique_tokens) * 0.3)
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


def extract_named_entities(comments):
    entities = set()
    for comment in comments:
        doc = nlp(comment)
        for ent in doc.ents:
            entities.add(ent.text)
    return list(entities)


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


def save_graph(G):
    nodes = G.nodes()
    edges = G.edges(data=True)
    with open(nodes_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["word"])
        for node in nodes:
            writer.writerow([node])
    with open(edges_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["word1", "word2", "weight"])
        for source, target, data in edges:
            writer.writerow([source, target, data["weight"]])


df = pd.read_json(input_path, orient="records")
comments = df[0]

tokens = []
for comment in comments:
    tokens.extend(preprocess_text(comment))
unique_tokens = list(set(tokens))

# Get tokens that appears more than n times in the comments
high_fequency_tokens = filter_frequent_words(tokens)
unique_tokens_high_frequency = list(set(high_fequency_tokens))

word_clusters = cluster_similar_words(unique_tokens)

graph = build_cooccurrence_graph(comments, unique_tokens_high_frequency)
filtered_graph = filter_graph(graph)

print(
    "Initial number of tokens: ",
    len(tokens),
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

visualize_graph_interactive(filtered_graph)
# Example: Display connections for the word "hate"
# save_graph(graph)
