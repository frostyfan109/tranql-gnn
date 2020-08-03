import stellargraph as sg
import numpy as np
from sklearn import feature_extraction
from tranql_jupyter import KnowledgeGraph

def make_features(k_graph, format):
    vec = feature_extraction.DictVectorizer(sparse=False, dtype=np.float32)
    nodes = format(k_graph)
    feature_vectors = vec.fit_transform(nodes)
    # Zip each node/feature_vector into (node, feature_vector)
    feature_zip = zip(k_graph.net.nodes, feature_vectors)

    return feature_zip

def make_dataset(k_graph, format=None):
    for e in k_graph.net.edges(data=True):
        # if "weight" in e[2]: del e[2]["weight"]
        # e[2]["type"] = np.array(e[2]["type"])
        e[2]["label"] = e[2]["type"][0]
        pass

    if format == None:
        dataset = sg.StellarGraph.from_networkx(k_graph.net)
    else:
        dataset = sg.StellarGraph.from_networkx(k_graph.net, node_features=make_features(k_graph, format))

    return dataset