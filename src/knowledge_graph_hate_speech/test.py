import spacy
import networkx as nx
from pyvis.network import Network

nlp = spacy.load("en_core_web_md")
G = nx.Graph()
window_size = 3

comment = "you imagine being so retarded. you should be sterilized."
grap_unclustered_output_path = "GK_graph_unclustered.html"


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


doc = nlp(comment)

result = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print(result)

reduced_tokens = set()
for token in result:
    reduced_tokens.add(token)
reduced_tokens = list(reduced_tokens)

for i in range(len(reduced_tokens)):
    for j in range(i + 1, min(i + window_size, len(reduced_tokens))):
        if G.has_edge(reduced_tokens[i], reduced_tokens[j]):
            G[reduced_tokens[i]][reduced_tokens[j]]["weight"] += 1
        else:
            G.add_edge(reduced_tokens[i], reduced_tokens[j], weight=1)

visualize_graph_interactive(G)
