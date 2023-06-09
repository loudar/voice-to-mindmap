import json
import math
import os
import sys
import time
from datetime import datetime

import Levenshtein as Levenshtein
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import nltk as nltk
import spacy
import babelnet as bn

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
    color_palette = cm.get_cmap('rainbow', len(categories))

    for i, category in enumerate(categories):
        color_mapping[category] = color_palette(i)

    # Draw nodes with labels and circles
    for n in G.nodes():
        x, y = pos[n]
        node_size = G.nodes[n]['size'] * min_dimension * 0.02 / max_node_size

        # if not planar:
        #     ellipse_width = node_size * (1 / ratio)
        #     ellipse_height = node_size
        #     ellipse = Ellipse((x, y), width=ellipse_width, height=ellipse_height, color='green', alpha=0.3)
        #     plt.gca().add_patch(ellipse)
        # else:
        #     circle = Ellipse((x, y), width=node_size, height=node_size, color='green', alpha=0.3)
        #     plt.gca().add_patch(circle)

        # Get the category of the node
        category = G.nodes[n].get('category')

        # Set the color of the node label based on the category
        label_color = 'black'
        if category:
            label_color = color_mapping[category]

        # Draw the node label with the specified color
        font_size = 8 + G.nodes[n]['size'] * 0.5
        plt.text(x, y, n, color=label_color, ha='center', va='center', fontsize=font_size)

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
    if not word:
        return 'unknown'

    word_lower = word.lower()
    cached_category = get_cached_word_categories(word_lower, language)
    if cached_category != '__not_cached__':
        print(f"Word category for '{word}' is cached as '{cached_category}'.")
        return cached_category

    # Query BabelNet
    print(f"Querying BabelNet for '{word}'.")
    babel_lang = None
    if language == 'en':
        babel_lang = bn.Language.EN
    elif language == 'de':
        babel_lang = bn.Language.DE
    else:
        Exception(f"Language {language} not supported.")

    synsets = bn.get_synsets(word, from_langs=[babel_lang], to_langs=[babel_lang])

    if not synsets:
        cache_word_categories(word, 'unknown', language)
        return 'unknown'

    word_category_map = {}
    for synset in synsets:
        raw_categories = synset.categories(babel_lang)
        string_categories = [category.category.lower() for category in raw_categories]
        lemma_lower = synset.main_sense_preferably_in(babel_lang).full_lemma.lower()
        if word_lower in string_categories:
            string_categories.remove(word_lower)

        if len(string_categories) == 0:
            continue

        word_category_map[lemma_lower] = string_categories

    if word_lower in word_category_map:
        closest = closest_match(word_lower, word_category_map[word_lower])
        cache_word_categories(word_lower, closest, language)
        print(f"Word category for '{word}' is '{closest}' (out of {word_category_map[word_lower]}).")
        return closest
    else:
        word_keys = word_category_map.keys()
        min_word = closest_match(word_lower, word_keys)

        if min_word:
            print(f"Word '{word}' not found in BabelNet. Closest match is '{min_word}' (out of {word_keys}).")
            closest = closest_match(word_lower, word_category_map[min_word])
            cache_word_categories(word_lower, closest, language)
            cache_word_categories(min_word, closest, language)
            print(f"Word category for closest match '{min_word}' is '{closest}' (out of {word_category_map[min_word]}).")
            return closest
        else:
            cache_word_categories(word, 'unknown', language)
            return 'unknown'


def closest_match(word, str_list):
    min_distance = 100
    min_word = None
    for w in str_list:
        distance = Levenshtein.distance(word, w)
        if distance < min_distance:
            min_distance = distance
            min_word = w

    return min_word


def cache_word_categories(word, category, language):
    filename = create_word_cache(language)
    with open(filename, 'r') as f:
        data = json.load(f)

    if word not in data:
        data[word] = category

    with open(filename, 'w') as f:
        json.dump(data, f)


def get_cached_word_categories(word, language):
    filename = create_word_cache(language)
    with open(filename, 'r') as f:
        data = json.load(f)

    if word in data:
        return data[word]
    else:
        return '__not_cached__'


def create_word_cache(language):
    folder = "cache"
    filename = f"{folder}/{language}.json"
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("{}")

    return filename

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
