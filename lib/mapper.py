import networkx as nx


def create_mind_map_force(proximity_links):
    G = nx.Graph()

    # Add nodes and edges to the graph
    add_nodes_and_edges_force(G, proximity_links)

    # Dictionary to hold subgraphs for each node
    subgraphs = {}

    # Create subgraphs for each node connected to nodes without any other connections
    for node in G.nodes:
        connected_nodes = set(G.neighbors(node))
        unconnected_nodes = set(G.nodes) - connected_nodes - {node}

        # Check if the node has connections only to unconnected nodes
        if len(connected_nodes) > 0 and len(unconnected_nodes) == 0:
            subgraph = G.subgraph(connected_nodes).copy()
            subgraphs[node] = subgraph

    # Generate positions for each subgraph using spring_layout
    all_positions = {}
    for node, subgraph in subgraphs.items():
        positions = nx.spring_layout(subgraph)
        subgraph_positions = {n: pos for n, pos in positions.items()}
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