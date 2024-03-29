import plotly.graph_objects as go
from lib.colors import generate_unique_color


def create_plot(G, subgraph_positions, live_mode=False, show_labels=True, title="Mind Map"):
    print("Creating plot...")
    # Create Plotly figure
    edge_traces = []

    category_colors = {}  # Dictionary to store unique colors per category

    values = dict(G.degree()).values()
    if len(values) == 0:
        max_degree = 0
    else:
        max_degree = max(values)  # Calculate the maximum degree in the graph

    max_weight = [edge[2]['weight'] for edge in G.edges(data=True) if 'weight' in edge[2]]
    min_weight = min(max_weight) if len(max_weight) > .5 else .5
    for edge in G.edges(data=True):
        try:
            x0, y0 = subgraph_positions[edge[0]]
            x1, y1 = subgraph_positions[edge[1]]
            weight = edge[2]['weight'] if 'weight' in edge[2] else 1
            weight = (weight - min_weight) / max(max_weight) * 5

            edge_trace = go.Scatter(
                x=[x0, (x0 + x1) / 2, x1, None],
                y=[y0, (y0 + y1) / 2, y1, None],
                line=dict(width=weight, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False,
            )

            edge_traces.append(edge_trace)
        except KeyError:
            print(f"Edge {edge} not found in subgraph_positions")
            continue

    node_mode = 'markers+text' if show_labels else 'markers'
    for node in G.nodes():
        category = G.nodes[node].get('category', 'default')  # Get the category property
        size = G.degree[node] / max_degree  # Calculate relative node size based on the degree
        size = max(size * 25, 1)

        if category not in category_colors:
            # Generate a unique color for the category with good contrast against white
            color = generate_unique_color(category_colors.values())
            category_colors[category] = color
        else:
            color = category_colors[category]

        node_text = node if live_mode else node + f" ({category})"
        try:
            node_trace = go.Scatter(
                x=[subgraph_positions[node][0]],
                y=[subgraph_positions[node][1]],
                mode=node_mode,
                marker=dict(
                    showscale=False,
                    color=f"rgb{color}",
                    size=size,
                    line=dict(width=2)
                ),
                text=[node],
                hovertext=[node_text],
                textfont=dict(color='black', size=10),
                hoverinfo='text',
                textposition="middle right",
            )

            edge_traces.append(node_trace)
        except KeyError:
            print(f"Node {node} not found in subgraph_positions")
            continue

    go_fig = go.Figure(data=edge_traces,
                       layout=go.Layout(
                           title=title,
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

    if live_mode:
        return go_fig
    else:
        go_fig.show()
