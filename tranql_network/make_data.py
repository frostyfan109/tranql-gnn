import numpy as np
import networkx as nx
from spektral.utils import nx_to_adj

def conv_list_attr(k_graphs, attr="type", nodes=True):
    """
    Go through and turn every node `type` attribute into a binary array where each element's index
    corresponds to its respective type.
        type_names is the aggregate of all types contained in the graphs of k_graphs.
            type_names = ["chemical_substance", "gene", "disease", "phenotypic_feature", "biological_entity", "genetic_condition"]
        Each node's `type` attribute is a binary array of length type_names. For each index/element, element is either 0 or 1,
        indicating whether or not the node has the type of type_names[index].
            n["type"] = [0, 0, 1, 0, 0, 1] means that n has the types "disease" and "genetic_condition"
    :param k_graphs: KnowledgeGraph[]
    :param attr: str
    :param nodes: bool
    :return: str[]
    """
    type_names = []
    if nodes:
        node_attrs = [
            n[1]["attr_dict"] for G in k_graphs
            for n in G.net.nodes(data=True)
        ]
    else:
        node_attrs = [
            n[2] for G in k_graphs
            for n in G.net.edges(data=True)
        ]

    for node in node_attrs:
        node_types = node[attr]

        for n_type in node_types:
            try:
                index = type_names.index(n_type)
            except:
                index = len(type_names)
                type_names.append(n_type)

    for node in node_attrs:
        node_types = node[attr]

        new_n_type = [0 for x in type_names]
        for n_type in node_types:
            new_n_type[type_names.index(n_type)] = 1
        node[attr] = new_n_type

    return type_names

def make_node_attributes(k_graphs, attr="type"):
    """
    Make node attributes using type as feature from graphs. Return list of shape (n_graphs x N x F) where F is feature length
    :return: np.float32[n_graphs]
    """
    nx_graphs = [k_graph.net for k_graph in k_graphs]
    x = []
    for G in nx_graphs:
        node_array = []
        for node in G.nodes(data=True):
            node_array.append(node[1]["attr_dict"][attr])
        x.append(np.float32(node_array))
    return x



def make_data(k_graphs):
    nx_graphs = [kg.net for kg in k_graphs]

    graph_adj = nx_to_adj(nx_graphs) # shape = n_graphs x N x N where N is max num nodes in graphs

    type_names = conv_list_attr(k_graphs)
    graph_node = make_node_attributes(k_graphs) # shape = n_graphs x N x F where F is feature length (num total types)

    graph_edge = None
    # Let's try to predict which reasoners nodes come from based on the types of the nodes
    # This isn't as simple as it seems at face value, because a node can be of multiple types
    # Still more or less just a simple probabilistic model
    reasoner_names = conv_list_attr(k_graphs, attr="reasoner")
    labels = make_node_attributes(k_graphs, attr="reasoner") # shape = n_graphs x N x n_categories

    return (
        graph_adj, (type_names, graph_node), graph_edge, (reasoner_names, labels)
    )

def make_masks(num_nodes, val=0.1, test=0.1):
    perm = np.random.permutation(num_nodes)

    end_val = int(num_nodes * val)
    val_mask = np.zeros(num_nodes, dtype=np.float32)
    val_mask[perm[0:end_val]] = 1

    end_test = end_val + int(num_nodes * test)
    test_mask = np.zeros(num_nodes, dtype=np.float32)
    test_mask[perm[end_val:end_test]] = 1

    train_mask = np.zeros(num_nodes, dtype=np.float32)
    train_mask[end_test:-1] = 1

    return (
        train_mask, val_mask, test_mask
    )