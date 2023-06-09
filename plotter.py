import math
import os
import sys
from datetime import datetime
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import nltk as nltk
import spacy
import babelnet

languages = ['en', 'de']
models = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}
selected_lang = 'de'

nlp = spacy.load(models[selected_lang])
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

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
        pos = nx.spring_layout(G, center=(0, 0), weight='weight', scale=1.0, iterations=50)
    except nx.NetworkXException as e:
        print(f"Planar layout not possible. Switching to shell layout: {e}")
        planar = False
        pos = nx.shell_layout(G, scale=min_dimension)

    max_node_size = max([G.nodes[n]['size'] for n in G.nodes()])

    # Create a color mapping dictionary for categories
    categories = set()
    for node in G.nodes():
        category = G.nodes[node].get('category')
        if category:
            categories.add(category)

    print(f"Categories: {categories}")
    color_mapping = {}
    color_palette = cm.get_cmap('hsv', len(categories))

    for i, category in enumerate(categories):
        color_mapping[category] = color_palette(i)

    # Draw nodes with labels and circles
    for n in G.nodes():
        x, y = pos[n]
        node_size = G.nodes[n]['size'] * min_dimension * 0.02 / max_node_size

        if not planar:
            ellipse_width = node_size * (1 / ratio)
            ellipse_height = node_size
            ellipse = Ellipse((x, y), width=ellipse_width, height=ellipse_height, color='green', alpha=0.3)
            plt.gca().add_patch(ellipse)
        else:
            circle = Ellipse((x, y), width=node_size, height=node_size, color='green', alpha=0.3)
            plt.gca().add_patch(circle)

        # Get the category of the node
        category = G.nodes[n].get('category')

        # Set the color of the node label based on the category
        label_color = 'black'
        if category:
            label_color = color_mapping[category]

        # Draw the node label with the specified color
        plt.text(x, y, n, color=label_color, ha='center', va='center', fontsize=8)

    # Draw edges
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    margin = 10
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='blue', arrows=True, arrowstyle='-|>', arrowsize=12,
                           min_source_margin=margin, min_target_margin=margin)

    # Adjust plot limits
    plt.axis('off')

    # Update the plot size
    size_factor = len(G.nodes()) / 10
    dpi = 100
    pixel_x = math.floor(fig_width * size_factor * dpi)
    pixel_y = int(fig_height * size_factor * dpi)
    max_pixels = 2000
    if pixel_x > max_pixels or pixel_y > max_pixels:
        print(f"Plot size exceeds maximum of 5000 x 5000 px. Scaling down.")
        reduction_factor = max_pixels / max(pixel_x, pixel_y)
        size_factor *= reduction_factor
    elif pixel_x < 500 or pixel_y < 500:
        print(f"Plot size is too small. Scaling up.")
        reduction_factor = 1500 / min(pixel_x, pixel_y)
        size_factor *= reduction_factor
    plt.gcf().set_size_inches(fig_width * size_factor, fig_height * size_factor)

    # Refresh the plot
    plt.draw()
    print(f"Saving plot to: {plot_file}")
    try:
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1)
        os.system(f"start {plot_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")


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



def get_word_category(word, language='en'):
    bn = babelnet.BabelNetAPI()

    # Query BabelNet
    synsets = bn.getSynsetIds(word, lang=language)

    # If synsets were found
    if synsets:
        # Get the first synset
        first_synset = bn.getSynset(synsets[0]['id'])
        # Get the main sense (most representative word)
        main_sense = first_synset.getMainSense()
        # Extract the part of speech from the main sense
        pos_tag = main_sense.split('#')[1][0]

        # Map POS tag to BabelNet tag
        if pos_tag == 'n':
            bn_tag = 'NOUN'
        elif pos_tag == 'v':
            bn_tag = 'VERB'
        elif pos_tag == 'a':
            bn_tag = 'ADJ'
        elif pos_tag == 'r':
            bn_tag = 'ADV'
        else:
            bn_tag = None

        # Return the category
        return bn_tag

    return None


def sort_nodes_by_weight(g):
    return sorted(g.nodes(), key=lambda node: g.nodes[node]['size'], reverse=True)


def create_mind_map(proximity_links):
    G = nx.Graph()
    tempnodes = {}
    tempedges = {}

    for word1, word2 in proximity_links:
        if word1 not in tempnodes:
            tempnodes[word1] = 1
        else:
            tempnodes[word1] += 1

        if word2 not in tempnodes:
            tempnodes[word2] = 1
        else:
            tempnodes[word2] += 1

        if word1 not in tempedges:
            tempedges[word1] = {}

        if word2 not in tempedges[word1]:
            if word1 == word2:
                continue
            tempedges[word1][word2] = 1
        else:
            tempedges[word1][word2] += 1

    # convert to list
    temp_edges = []
    for node in tempedges:
        for edge in tempedges[node]:
            temp_edges.append([node, edge, tempedges[node][edge]])

    # sort and get top 100
    sort_temp_edges = sorted(temp_edges, key=lambda x: x[2], reverse=True)
    # top_10_percent_count = int(len(sort_temp_edges) * 0.1)
    top_count = 50
    top_list = sort_temp_edges[:top_count]

    # convert back to dict
    tempedges = {}
    for edge in sort_temp_edges:
        tempedges[edge[0]] = {}
        tempedges[edge[0]][edge[1]] = edge[2]

    for edge in top_list:
        if edge[0] not in G.nodes():
            G.add_node(edge[0], size=tempnodes[edge[0]], category=get_word_category(edge[0], selected_lang))

        if edge[1] not in G.nodes():
            G.add_node(edge[1], size=tempnodes[edge[1]], category=get_word_category(edge[1], selected_lang))

        G.add_edge(edge[0], edge[1], weight=edge[2])

    print(f"Added {len(G.nodes())} nodes and {len(G.edges())} edges to graph.")

    return G


def main():
    with open(source_file, 'r', encoding='windows-1252') as f:
        text = f.read()

    update_plot(text)


main()
