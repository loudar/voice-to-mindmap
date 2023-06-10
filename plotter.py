import json
import os
import sys
from datetime import datetime
import Levenshtein as Levenshtein
import matplotlib.pyplot as plt
import networkx as nx
import nltk as nltk
import spacy
import numpy as np
import plotly.graph_objects as go

try:
    import babelnet as bn
except RuntimeError as e_babelnet:
    print("Babelnet API key is not valid at the moment.")

languages = ['en', 'de']
models = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}
selected_lang = 'de'

try:
    nlp = spacy.load(models[selected_lang])
except OSError as e:
    print(f"Error loading model: {e} \nPlease run 'python -m spacy download {models[selected_lang]}'")
    sys.exit(1)

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
    G, subgraph_positions = create_mind_map(logical_links)
    create_plot(G, subgraph_positions)


def extract_logical_links(text):
    doc = nlp(text)
    logical_links = []

    for sentence in doc.sents:
        subjects = []
        for token in sentence:
            if token.pos_ == 'NOUN' or token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                subjects.append((token.text, token.pos_))  # include the token's POS as the category
            elif token.pos_ == 'ADJ':
                subjects.append((token.text, token.pos_))  # include the token's POS as the category

        for i in range(len(subjects) - 1):
            logical_links.append({
                'source': subjects[i][0],
                'target': subjects[i + 1][0],
                'source_category': get_word_category(subjects[i][0], selected_lang),
                'target_category': get_word_category(subjects[i + 1][0], selected_lang),
            })

    return logical_links


def get_word_category(word, language='en'):
    if not word:
        return 'unknown'

    word_lower = word.lower()
    cached_category = get_cached_word_categories(word_lower, language)
    if cached_category != '__not_cached__':
        print(f"Cached category for '{word}' is '{cached_category}'.")
        return cached_category
    if cached_category != '__error__':
        print(f"Querying category for '{word}' returned an error last time.")
        return 'unknown'

    # Query BabelNet
    print(f"Querying BabelNet for '{word}'.")
    babel_lang = None
    if language == 'en':
        babel_lang = bn.Language.EN
    elif language == 'de':
        babel_lang = bn.Language.DE
    else:
        Exception(f"Language {language} not supported.")

    try:
        synsets = bn.get_synsets(word, from_langs=[babel_lang], to_langs=[babel_lang])

        if not synsets:
            print(f"No synsets found for '{word}'.")
            cache_word_categories(word_lower, 'unknown', language)
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
                print(
                    f"Word category for closest match '{min_word}' is '{closest}' (out of {word_category_map[min_word]}).")
                return closest
            else:
                print(f"No closest match found for '{word}' (in {word_keys}).")
                cache_word_categories(word_lower, 'unknown', language)
                return 'unknown'
    except RuntimeError as e_babel:
        print(f"Error querying BabelNet for '{word}': {e_babel}")
        cache_word_categories(word_lower, '__error__', language)
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


def add_nodes_and_edges(G, proximity_links):
    edge_weights = {}
    node_categories = {}

    # First, create all edges and their weights without adding them to the graph.
    for link in proximity_links:
        edge = (link['source'], link['target'])
        if edge in edge_weights:
            edge_weights[edge] += 1
        else:
            edge_weights[edge] = 1
        node_categories[link['source']] = link['source_category']
        node_categories[link['target']] = link['target_category']

    # Then, sort edges by weight and select top 50 or fewer if total number is less than 50.
    sorted_edges = sorted(edge_weights.items(), key=lambda item: item[1], reverse=True)
    top_edges = sorted_edges[:50] if len(sorted_edges) > 50 else sorted_edges

    # Now, add the top edges and their nodes to the graph.
    for edge, weight in top_edges:
        source, target = edge
        if source not in G.nodes:
            G.add_node(source, category=node_categories[source],
                       size=len([e for e in edge_weights if e[0] == source or e[1] == source]))
        if target not in G.nodes:
            G.add_node(target, category=node_categories[target],
                       size=len([e for e in edge_weights if e[0] == target or e[1] == target]))
        G.add_edge(source, target, weight=weight)


def create_mind_map(proximity_links, min_distance=10):
    G = nx.Graph()

    # Add nodes and edges to the graph
    add_nodes_and_edges(G, proximity_links)

    # Dictionary to hold subgraphs for each category
    subgraphs = {}

    # Create subgraphs for each category
    for node, data in G.nodes(data=True):
        category = data['category']
        if category not in subgraphs:
            subgraphs[category] = G.subgraph([node]).copy()
        else:
            subgraphs[category].add_node(node)

    # Generate positions for each subgraph
    subgraph_positions = {}

    for i, (category, subgraph) in enumerate(subgraphs.items()):
        # Use spring layout to position nodes within each subgraph
        positions = nx.spring_layout(subgraph, scale=1.0)
        # Adjust positions so that subgraphs are separated
        for pos in positions.values():
            pos[0] += i * 10
        subgraph_positions.update(positions)

    # Ensure minimum distance within each category
    for category, subgraph in subgraphs.items():
        nodes = list(subgraph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pos1 = np.array(subgraph_positions[nodes[i]])
                pos2 = np.array(subgraph_positions[nodes[j]])
                # Calculate distance
                distance = np.linalg.norm(pos1 - pos2)
                if distance < min_distance:
                    # Calculate direction vector
                    direction = (pos2 - pos1) / distance
                    # Move nodes apart
                    shift = (min_distance - distance) / 2
                    subgraph_positions[nodes[i]] -= direction * shift
                    subgraph_positions[nodes[j]] += direction * shift

    return G, subgraph_positions


import plotly.graph_objects as go

def create_plot(G, subgraph_positions):
    # Create Plotly figure
    edge_traces = []

    for edge in G.edges(data=True):
        x0, y0 = subgraph_positions[edge[0]]
        x1, y1 = subgraph_positions[edge[1]]
        weight = edge[2]['weight'] if 'weight' in edge[2] else 1

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=weight, color='#888'),  # set line width to weight
            hoverinfo='text',
            hovertemplate=f"{edge[0]} - {edge[1]}<br>Weight: {weight}<extra></extra>",
            text=[f"{edge[0]} - {edge[1]}"],
            hovertext=[f"{edge[0]} - {edge[1]}"]
        )

        edge_traces.append(edge_trace)

    node_trace = go.Scatter(
        x=[subgraph_positions[node][0] for node in G.nodes()],
        y=[subgraph_positions[node][1] for node in G.nodes()],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        ),
        text=list(G.nodes())
    )

    go_fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Network graph made with Python',
                           titlefont=dict(size=16),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )

    go_fig.show()


def main():
    with open(source_file, 'r', encoding='windows-1252') as f:
        text = f.read()

    update_plot(text)


main()
