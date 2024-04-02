import networkx as nx

def edges_to_be_cut(forest):
    somas = [node for node, deg in forest.degree() if deg >= 4]
    print(len(somas))
    ends = [node for node, deg in forest.degree() if deg == 1]
    print(len(ends))


if __name__ == '__main__':
    from ntools.dbio import read_nodes, read_edges
    db_path = 'tests/z002_labeled.db'
    node_id = 19604
    nodes = read_nodes(db_path)
    edges = read_edges(db_path)
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['nid'], nid = node['nid'],coord = node['coord'], nbrs = node['nbrs'], checked = node['checked'])
    for edge in edges:
        G.add_edge(edge['src'],edge['des'],creator = edge['creator'])
    
    forest = nx.node_connected_component(G,node_id)
    forest = G.subgraph(forest)
    edges_to_be_cut(forest)
