import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.optimize import minimize

def create_mind_map_force(proximity_links):
    G = nx.Graph()

    # Add nodes and edges to the graph
    add_nodes_and_edges_force(G, proximity_links)

    # Dictionary to hold subgraphs for each category
    subgraphs = {}

    # Create subgraphs for each category
    for node, data in G.nodes(data=True):
        category = data['category']
        if category not in subgraphs:
            subgraphs[category] = G.subgraph([node]).copy()
        else:
            subgraphs[category].add_node(node)

    # Generate positions for each category
    category_positions = nx.circular_layout(list(subgraphs.keys()), scale=100.0)

    # Dictionary to hold all positions
    all_positions = {}

    # Generate positions for each subgraph using force-directed layout
    i = 0
    for category, subgraph in subgraphs.items():
        i += 1
        print(f"Generating positions for category '{category}' ({i}/{len(subgraphs.items())})")
        num_nodes = subgraph.number_of_nodes()

        # Create initial random positions for nodes within each subgraph
        init_positions = np.random.rand(num_nodes, 2) * 10.0

        # Calculate the weight matrix based on edge weights
        weight_matrix = np.zeros((num_nodes, num_nodes))
        nodes = list(subgraph.nodes)
        for i, j, data in subgraph.edges(data=True):
            weight_matrix[nodes.index(i)][nodes.index(j)] = data['weight']
            weight_matrix[nodes.index(j)][nodes.index(i)] = data['weight']

        # Define the objective function for force-directed layout
        def objective(positions):
            positions = positions.reshape((num_nodes, 2))
            dist_matrix = distance_matrix(positions, positions)

            # Define minimum distance
            min_dist = 5.0

            # Increase repulsion force for nodes closer than min_dist
            repulsion = np.sum(np.divide(1, np.where(dist_matrix > min_dist, dist_matrix, min_dist)))

            # Calculate attraction force, excluding division by zero
            dist_matrix_nonzero = np.where(dist_matrix != 0, dist_matrix, 1e-10)
            attraction = np.sum(np.divide(weight_matrix, dist_matrix_nonzero))

            # Calculate central force, which is proportional to the total weight of incident edges
            center = np.mean(positions, axis=0)
            central_force = np.sum(np.linalg.norm(positions - center, axis=1) * np.sum(weight_matrix, axis=1))

            return repulsion - attraction + central_force

        # Perform optimization to minimize the objective function
        result = minimize(objective, init_positions.flatten())

        # Update positions with optimized values
        positions = result.x.reshape((num_nodes, 2))

        # Scale positions based on the inverse of their weight to position higher weight nodes closer to the center
        positions = positions / np.array([G.nodes[node]['size'] for node in nodes]).reshape(-1, 1)

        # Adjust positions to fit the mind map
        positions = positions * 10.0 + category_positions[category]

        # Create a dictionary of positions for the subgraph
        subgraph_positions = {node: tuple(position) for node, position in zip(subgraph.nodes, positions)}
        all_positions.update(subgraph_positions)

    return G, all_positions


def add_nodes_and_edges_force(G, proximity_links):
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
