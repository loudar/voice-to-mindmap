import sys
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import spacy
from matplotlib.patches import Ellipse

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

if len(sys.argv) > 1:
    source_file = sys.argv[1]
else:
    source_file = "transcripts/transcript_latest.txt"

# Create a figure and axes for the plot
fig, ax = plt.subplots()


def update_plot(text):
    # Extract logical links from the new text
    logical_links = extract_logical_links(text)

    # Create the mind map using the accumulated logical links
    G = create_mind_map(logical_links)

    # Clear the current plot
    plt.clf()

    # Set background color
    plt.gca().set_facecolor('white')

    # Calculate scaling factor based on the current figure size
    fig_width, fig_height = plt.gcf().get_size_inches()

    # Determine the smaller dimension
    min_dimension = min(fig_width, fig_height)
    ratio = fig_width / fig_height

    # Scale and adjust node positions based on the scaling factor
    planar = True
    try:
        pos = nx.planar_layout(G, center=(0, 0))
    except nx.NetworkXException as e:
        print(f"Planar layout not possible. Switching to shell layout: {e}")
        planar = False
        pos = nx.shell_layout(G, scale=min_dimension)

    # Draw nodes with labels and circles
    for n in G.nodes():
        x, y = pos[n]
        node_size = G.nodes[n]['size'] * min_dimension * 0.01

        # Calculate the width and height of the ellipse
        ellipse_width = node_size * (1 / ratio)
        ellipse_height = node_size

        # Create and add the circle patch using the scaled node size
        if not planar:
            ellipse = Ellipse((x, y), width=ellipse_width, height=ellipse_height, color='green', alpha=0.3)
            plt.gca().add_patch(ellipse)

        # Draw the node label
        plt.text(x, y, n, color='black', ha='center', va='center', fontsize=8)

    # Draw edges
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='blue', arrows=True, arrowstyle='-|>', arrowsize=12)

    # Adjust plot limits
    plt.axis('off')

    # Update the plot size
    plt.gcf().set_size_inches(fig_width, fig_height)

    # Refresh the plot
    plt.draw()
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(latest_plot_file, bbox_inches='tight', pad_inches=0.1)
    plt.pause(0.001)


def extract_logical_links(text):
    doc = nlp(text)
    logical_links = []

    for sentence in doc.sents:
        subjects = []
        for token in sentence:
            if token.pos_ == 'NOUN' or token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                subjects.append(token.text)
            elif token.pos_ == 'ADJ':
                subjects.append(token.text)

        for i in range(len(subjects) - 1):
            logical_links.append((subjects[i], subjects[i + 1]))

    return logical_links


def create_mind_map(proximity_links):
    G = nx.Graph()

    for word1, word2 in proximity_links:
        if word1 not in G.nodes():
            G.add_node(word1, size=1)
        else:
            G.nodes[word1]['size'] += 1

        if word2 not in G.nodes():
            G.add_node(word2, size=1)
        else:
            G.nodes[word2]['size'] += 1

        if not G.has_edge(word1, word2):
            if word1 == word2:
                continue
            G.add_edge(word1, word2, weight=1)
        else:
            G[word1][word2]['weight'] += 1

    return G


def main():
    with open(source_file, 'r', encoding='windows-1252') as f:
        text = f.read()

    update_plot(text)


main()
