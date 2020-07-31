import json
from spektral.utils import nx_to_numpy
from tranql_jupyter import KnowledgeGraph

def load_knowledge_graph(f):
    return KnowledgeGraph(json.load(f))


def knowledge_graphs_to_spektral(knowledge_graphs, n_keys=None, e_keys=None):
    nx_graphs = [x.net for x in knowledge_graphs]
    for net in nx_graphs:
        nodes = net.nodes(data=True)
        for node in nodes:
            """
            Go through each node in the graph and turn it from
            {"attr_dict": attrs} -> {**attrs}
            Note: node is a tuple, so its elements can't be reassigned
            """
            properties = node[1]
            attrs = properties["attr_dict"]
            del properties["attr_dict"]
            properties.update(attrs)

    return to_spektral(nx_graphs, n_keys=n_keys, e_keys=e_keys)


""" For data representation in Spektral,
    refer to: https://graphneural.network/data/ """
def to_spektral(knowledge_graphs, n_keys=None, e_keys=None):
    """
    Note: knowledge_graphs may be a single NetworkX graph rather than a list.
          If this is the case, the first dimension n_graphs is dropped.
    
    graph_adj = A = Adjacency matrix of shape (n_graphs, N, N) where N is the
    number of nodes
    
    node_attr_matrix = X = Node attributes matrix of shape (n_graphs, N, F)
    where F is the size of the node attributes
    
    edge_attr_matrix = E = Edge attributes matrix of shape (n_graphs, n_edges, S)
    where S is the size of the edge attributes
    """
    graph_adj, node_attr_matrix, edge_attr_matrix = nx_to_numpy(knowledge_graphs,
                                                                nf_keys=n_keys,
                                                                ef_keys=e_keys)
    
    return (graph_adj, node_attr_matrix, edge_attr_matrix)
