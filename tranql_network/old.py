import numpy as np
from tranql_jupyter import KnowledgeGraph
from .dataset import knowledge_graphs_to_spektral
from spektral.utils import label_to_one_hot, nx_to_adj

k_graphs = (KnowledgeGraph.mock1(), KnowledgeGraph.mock2())

type_names = []
node_attrs = [
    n[1]["attr_dict"] for G in k_graphs
    for n in G.net.nodes(data=True)
]

for node in node_attrs:
    node_types = node["type"]
    # For now just use the first type the node has until can figure
    # out how to do something like "multi-hot encoding"
    selected_type = node_types[0]
    try:
        index = type_names.index(selected_type)
    except:
        index = len(type_names)
        type_names.append(selected_type)
    node["type"] = index

nx_graphs = [kg.net for kg in k_graphs]
graph_adj = nx_to_adj(nx_graphs)

graph_adj, graph_node, graph_edge = knowledge_graphs_to_spektral(k_graphs,
                                                                 n_keys=["type"])

uniq_node = np.unique([v for x in graph_node for v in np.unique(x)])
node = [label_to_one_hot(x, uniq_node) for x in graph_node]

N = node[0].shape[-2]  # Number of nodes in the graphs
F = node[0].shape[-1]  # Original feature dimensionality
n_classes = label.shape[-1]  # Number of classes
l2_reg = 5e-4  # Regularization rate for l2
learning_rate = 1e-4  # Learning rate for Adam
epochs = 50  # Number of training epochs
batch_size = 32  # Batch size
es_patience = 5  # Patience for early stopping
es_delta = 0.001  # Min Delta for early stopping
