from pathlib import Path
import pandas as pd
from pyvis.network import Network
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from rdflib import Graph, URIRef, Literal, Namespace
from urllib.parse import quote  # Import the quote function for URI encoding

tqdm.pandas()

root_dir = Path(__file__).resolve().parent
input_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships_third_version.json"
)
output_kg_visualization_path = root_dir / "knowledge_graph_third_version.html"
output_kg_turtle_path = root_dir / "knowledge_graph_turtle.ttl"

data = {"extracted_relationships": []}


def load_dataset(dataset_path):
    """
    Loads the given dataset by using pandas library.
    Accepts dataset in json format.
    """
    df = pd.read_json(dataset_path, orient="records")
    return df


# Load dataframe
df = load_dataset(input_path)

for index, row in tqdm(df.iterrows(), total=len(df)):
    data["extracted_relationships"].append(row["extracted_relationships"])

relationships = pd.DataFrame(data)

# Count the frequency of each triple
triple_counts = defaultdict(int)

for rel_list in df["extracted_relationships"]:
    if rel_list is None:
        continue
    for triple in rel_list:
        if triple is None:
            continue
        # Skip triples that don't have the required keys
        if not all(key in triple for key in ["subject", "predicate", "object"]):
            continue
        # Skip triples with None values
        if (
            triple["subject"] is None
            or triple["predicate"] is None
            or triple["object"] is None
        ):
            continue
        key = (triple["subject"], triple["predicate"], triple["object"])
        triple_counts[key] += 1

# Convert to a list of triples with weights
weighted_triples = [(s, p, o, w) for (s, p, o), w in triple_counts.items()]

num_distinct_triples = len(triple_counts)
print(f"Total distinct triples: {num_distinct_triples}")
distinct_nodes = set()

for subject, predicate, obj in triple_counts.keys():
    distinct_nodes.add(subject)
    distinct_nodes.add(obj)

num_distinct_nodes = len(distinct_nodes)
print(f"Total distinct nodes: {num_distinct_nodes}")


# Create an RDF graph
g = Graph()

# Define a namespace for your graph
ns = Namespace("http://hate_speech_detection.org")

# Create a directed graph for visualization
G = nx.DiGraph()

# Add weighted triples to the RDF graph and the visualization graph
for s, p, o, w in weighted_triples:
    # Skip triples with None values (just in case)
    if s is None or p is None or o is None:
        continue

    # Encode the subject, predicate, and object to make them valid URIs
    subject = URIRef(ns + "/" + quote(s.encode("utf-8")))  # Encode string to bytes
    predicate = URIRef(ns + "/" + quote(p.encode("utf-8")))  # Encode string to bytes
    obj = URIRef(ns + "/" + quote(o.encode("utf-8")))  # Encode string to bytes
    connection = URIRef(
        ns + "/" + quote(s.encode("utf-8") + p.encode("utf-8") + o.encode("utf-8"))
    )  # Encode string to bytes
    weight = URIRef(ns + "/" + quote(Literal(w)))

    # Add the triple to the RDF graph
    g.add((subject, predicate, obj))

    # Optionally, add the weight as a separate triple
    g.add((connection, URIRef(ns + "/" + quote("weight")), weight))

    # Add nodes and edges to the visualization graph
    G.add_node(s, title=s)
    G.add_node(o, title=o)
    G.add_edge(s, o, title=p, weight=w)

# Serialize the RDF graph to a file (e.g., in Turtle format)
g.serialize(destination=output_kg_turtle_path, format="turtle")

# Visualize the graph using pyvis
net = Network(notebook=True, directed=True)
net.from_nx(G)

# Save the visualization to an HTML file
net.show(str(output_kg_visualization_path))  # Convert Path object to string
