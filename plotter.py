import sys
from datetime import datetime
import matplotlib.pyplot as plt
import nltk as nltk
import plotly.graph_objects as go

from lib.colors import generate_unique_color
from lib.mapper import create_mind_map_force
from lib.text_processing import extract_logical_links

languages = ['en', 'de']
selected_lang = 'de'

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
    logical_links = extract_logical_links(text, selected_lang)

    # Create the mind map using the accumulated logical links
    G, subgraph_positions = create_mind_map_force(logical_links)
    create_plot(G, subgraph_positions)


def create_plot(G, subgraph_positions):
    print("Creating plot...")
    # Create Plotly figure
    edge_traces = []

    category_colors = {}  # Dictionary to store unique colors per category

    for edge in G.edges(data=True):
        x0, y0 = subgraph_positions[edge[0]]
        x1, y1 = subgraph_positions[edge[1]]
        weight = edge[2]['weight'] if 'weight' in edge[2] else 1

        edge_trace = go.Scatter(
            x=[x0, (x0 + x1) / 2, x1, None],
            y=[y0, (y0 + y1) / 2, y1, None],
            line=dict(width=weight, color='#888'),
            hoverinfo='text',
            hovertext=f"{edge[0]} -> {edge[1]} ({weight})",
            mode='lines',
            showlegend=False,
        )

        edge_traces.append(edge_trace)

    for node in G.nodes():
        category = G.nodes[node].get('category', 'default')  # Get the category property
        size = G.degree[node]  # Calculate node size based on the count of connected nodes

        if category not in category_colors:
            # Generate a unique color for the category with good contrast against white
            color = generate_unique_color(category_colors.values())
            category_colors[category] = color
        else:
            color = category_colors[category]

        node_trace = go.Scatter(
            x=[subgraph_positions[node][0]],
            y=[subgraph_positions[node][1]],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=f"rgb{color}",
                size=size * 10,
                line=dict(width=2)
            ),
            text=[node + f" ({category})"],
        )

        edge_traces.append(node_trace)

    go_fig = go.Figure(data=edge_traces,
                       layout=go.Layout(
                           title='Generated Mind Map',
                           titlefont=dict(size=16),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           annotations=[dict(
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
