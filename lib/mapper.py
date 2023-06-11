import networkx as nx
import numpy as np


def create_mind_map_force(proximity_links):
    G = nx.Graph()

    # Add nodes and edges to the graph
    add_nodes_and_edges(G, proximity_links)

    # List to hold subgraphs for nodes with the threshold condition
    subgraphs = []

    # Define the threshold for connected nodes
    threshold = 2

    # Create subgraphs for nodes with the threshold or more connected nodes
    for node in G.nodes:
        connected_nodes = set(G.neighbors(node))

        # Check if the node meets the threshold condition
        if len(connected_nodes) >= threshold:
            subgraph = G.subgraph(connected_nodes).copy()
            subgraphs.append(subgraph)

    # Generate positions for each subgraph using spring_layout
    all_positions = {}
    for subgraph in subgraphs:
        positions = nx.shell_layout(subgraph, scale=0.3)
        for node in subgraph.nodes:
            all_positions[node] = positions[node]

    if len(all_positions) == 0:
        return G, all_positions

    all_positions = nx.spring_layout(G, pos=all_positions, iterations=50)

    # Call move_subgraphs function to adjust positions
    all_positions = move_subgraphs(G, all_positions)

    return G, all_positions


def move_subgraphs(G, all_positions):
    # Find the subgraphs with the largest number of nodes
    largest_subgraphs = max(nx.connected_components(G), key=len)

    # Calculate the center position
    center = np.mean(np.array(list(all_positions.values())), axis=0)

    # Move the larger subgraphs closer to the center position
    for node in largest_subgraphs:
        pos = all_positions[node]
        direction = center - pos
        new_pos = pos + direction * 0.1  # Adjust the scaling factor (0.1) as desired
        all_positions[node] = new_pos

    return all_positions


def add_nodes_and_edges(G, proximity_links):
    edge_weights = {}
    node_categories = {}

    # First, create all edges and their weights without adding them to the graph.
    for link in proximity_links:
        edge = (link['source'], link['target'])
        if edge in edge_weights:
            edge_weights[edge] += link['weight']
        else:
            edge_weights[edge] = link['weight']

        node_categories[link['source']] = link['source_category']
        node_categories[link['target']] = link['target_category']

    # Then, sort edges by weight and select top 50 or fewer if the total number is less than 50.
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
