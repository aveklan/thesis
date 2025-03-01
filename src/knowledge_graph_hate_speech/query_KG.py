from pathlib import Path
from rdflib import Graph, URIRef, Literal, Namespace
import urllib.parse

root_dir = Path(__file__).resolve().parent
input_kg_turtle_path = root_dir / "knowledge_graph_turtle.ttl"

# Define your namespace
ns = Namespace("http://hate_speech_detection.org/")


def shorten_uri(uri):
    """Removes namespace and decodes URL-encoded characters."""
    # Remove namespace if it starts with it
    if uri.startswith(ns):
        uri = uri[len(ns) :]

    # Decode URL-encoded characters (e.g., %20 -> space)
    return urllib.parse.unquote(uri)


loaded_g = Graph()
loaded_g.parse(input_kg_turtle_path, format="turtle")

query = """
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
  FILTER (
    CONTAINS(LCASE(STR(?s)), "retard") ||
    CONTAINS(LCASE(STR(?p)), "retard") ||
    CONTAINS(LCASE(STR(?o)), "retard")
  )
}
"""

# Run the query
results = loaded_g.query(query)

# Print results
for s, p, o in results:
    s = shorten_uri(s)
    p = shorten_uri(p)
    o = shorten_uri(o)
    if p == "weight":

        print(s, o)
    else:
        print(s, " ", p, " ", o)
