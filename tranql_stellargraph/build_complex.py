import numpy as np
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import KGTripleGenerator
from stellargraph.layer import ComplEx
from tensorflow import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
from dataset import make_dataset

""" ComplEx does not use node features """

MODEL_NAME = "tranql-complex-model"

train_size = 0.8
test_size = 0.2

# dataset, edge_train, edge_test, edge_val = sg.datasets.WN18().load()

EPOCHS = 100
EMBEDDING_DIMENSION = 200
NEGATIVE_SAMPLES = 10
LEARNING_RATE = 1E-3
PATIENCE = 10


def get_dataset(kg):
    return make_dataset(kg)


def make_model(dataset):
    edge_df = pd.DataFrame([
        {
            "source": e[0],
            "target": e[1],
            "label": e[2]["label"]
        } for e in dataset.to_networkx().edges(data=True)
    ])

    edge_train, edge_test = model_selection.train_test_split(
        edge_df, train_size=train_size, test_size=test_size
    )
    edge_train, edge_val = model_selection.train_test_split(
        edge_train, test_size=test_size
    )

    gen = KGTripleGenerator(
        dataset, batch_size=max(len(edge_train) // 100, 1) # make sure not to give batch size of 0
    )
    complex = ComplEx(
        gen,
        embedding_dimension=EMBEDDING_DIMENSION,
        embeddings_regularizer=l2(1E-7)
    )
    inp, out = complex.in_out_tensors()

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss=BinaryCrossentropy(from_logits=True),
        metrics=[BinaryAccuracy(threshold=0.0)]
    )

    train_gen = gen.flow(
        edge_train, negative_samples=NEGATIVE_SAMPLES, shuffle=True
    )
    valid_gen = gen.flow(
        edge_val, negative_samples=NEGATIVE_SAMPLES
    )

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    history = model.fit(
        train_gen, validation_data=valid_gen, epochs=EPOCHS, callbacks=[es]
    )

    sg.utils.plot_history(history)
    plt.show()

    return model


def make_type_predicate_mappings(k_graph):
    """ Map valid edge predicates between node types """
    """ E.g. mappings['chemical_substance']['gene'] = ['interacts_with', 'affects_activity_of', ...] """
    mappings = {}
    for edge in k_graph.net.edges(keys=True):
        n0, n1, predicate = edge
        n0_t = k_graph.net.nodes[n0]["attr_dict"]["type"]
        n1_t = k_graph.net.nodes[n1]["attr_dict"]["type"]
        for t0 in n0_t:
            for t1 in n1_t:
                if not t0 in mappings: mappings[t0] = {}
                if not t1 in mappings[t0]: mappings[t0][t1] = []
                if not predicate in mappings[t0][t1]: mappings[t0][t1].append(predicate)
    return mappings

def make_type_mappings(k_graph):
    """ Group nodes by their types """
    """ E.g. mappings['chemical_substance'] = ['CHEBI:X', 'CHEBI:Y', ...] """
    mappings = {}
    for node in k_graph.net.nodes(data=True):
        for type in node[1]["attr_dict"]["type"]:
            if not type in mappings: mappings[type] = []
            mappings[type].append(node[0])
    return mappings

def predict_edge(model, dataset, k_graph, edges, show_all=False):
    df = pd.DataFrame([
        {
            "source": e[0],
            "target": e[1],
            "label": e[2]
        } for e in edges
    ])
    gen = KGTripleGenerator(
        dataset,
        batch_size=10
    )
    flow = gen.flow(df)
    predictions = model.predict(flow)
    predictions = [prediction[0] for prediction in predictions]
    threshold = 0.0  # greater than 0.5 to be considered strongly predicted
    positive_predictions = [p for p in predictions if p > threshold]
    if show_all:
        positive_predictions = predictions

    if len(positive_predictions) == 0:
        src, dst = edges[0][0], edges[0][1]
        print(f"No edge predicted between {src} and {dst} (real={k_graph.net.has_edge(src, dst)}).")
    else:
        for i, prediction in enumerate(positive_predictions):
            src, dst, pred = edges[i]
            print(f"Edge {src}-[{pred}]->{dst} predicted ({prediction}) (real={k_graph.net.has_edge(src, dst, pred)})")


def main2():
    from tranql_jupyter import KnowledgeGraph
    k_graph = KnowledgeGraph.mock("mock3.json")
    dataset = get_dataset(k_graph)
    model = make_model(dataset)


def main():
    from random import randrange, choice
    from tranql_jupyter import KnowledgeGraph

    k_graph = KnowledgeGraph.mock("mock1.json")
    dataset = get_dataset(k_graph)
    model = make_model(dataset)

    num_edges = 2
    num_real_edges = 0

    net_nodes = list(k_graph.net.nodes)
    net_edges = list(k_graph.net.edges)
    """
    Get a map of valid predicates between the graph's node types
    Used for generating meaningful random edges for the model to process
        (because obviously the model isn't going to predict that the edge
         chemical_substance-[increases_synthesis_of]->phenotypic_feature
         exists because the predicate is nonsense in that context)
    """
    type_predicate_mappings = make_type_predicate_mappings(k_graph)
    type_mappings = make_type_mappings(k_graph)

    for i in range(num_edges):
        t0 = choice(list(type_predicate_mappings.keys()))
        t1 = choice(list(type_predicate_mappings[t0].keys()))
        # pred = choice(type_predicate_mappings[t0][t1])
        n0 = choice(type_mappings[t0])
        n1 = choice(type_mappings[t1])
        edges = [
            (n0, n1, predicate) for predicate in type_predicate_mappings[t0][t1]
        ]
        predict_edge(model, dataset, k_graph, edges)


if __name__ == "__main__":
    # main()
    main2()
