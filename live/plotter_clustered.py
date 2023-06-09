from datetime import datetime

import matplotlib.pyplot as plt
from netgraph import Graph
import spacy

languages = ['en', 'de']
models = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}
selected_lang = 'de'

nlp = spacy.load(models[selected_lang])

plot_folder = "maps/"
plot_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
plot_file = f"{plot_folder}mindmap_{plot_id}.svg"
latest_plot_file = f"{plot_folder}mindmap_latest.svg"

source_file = "transcripts/transcript_latest.txt"

# Read the transcription file
with open(source_file, 'r') as file:
    doc = nlp(file.read())

# Create a list of tuples for each sentence in the text, where each tuple contains the source node,
# target node, and edge weight (number of times the connection occurs)
edges = [(str(sent.root.head), str(sent.root), 1) for sent in doc.sents]

# Extract unique nodes
nodes = list(set([node for edge in edges for node in edge[:2]]))

# Create dictionaries to hold node and edge labels
node_labels = {node: f"Node {node}" for node in nodes}
edge_labels = {(edge[0], edge[1]): f"Edge {edge[0]}-{edge[1]}" for edge in edges}

# Customize nodes and edges (you can change these to suit your needs)
node_colors = {node: 'red' for node in nodes}
edge_colors = {(edge[0], edge[1]): 'blue' for edge in edges}

# Draw the graph
g = Graph(edges, node_labels=node_labels, edge_labels=edge_labels, node_color=node_colors, edge_color=edge_colors)
plt.show()
