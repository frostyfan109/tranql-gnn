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
from tranql_jupyter import KnowledgeGraph
from tranql_stellargraph.dataset import make_dataset

k_graph = KnowledgeGraph.mock("mock1.json")
dataset = make_dataset(k_graph)

""" ComplEx does not use node features """

MODEL_NAME = "tranql-complex-model"

train_size = 0.8
test_size = 0.2

edge_df = pd.DataFrame([
    {
        "source": e["source_id"],
        "target": e["target_id"],
        "label": e["label"]
    } for e in k_graph.build_knowledge_graph()["edges"]
])

edge_train, edge_test = model_selection.train_test_split(
    edge_df, train_size=train_size, test_size=test_size
)
edge_train, edge_val = model_selection.train_test_split(
    edge_train, test_size=test_size
)

# dataset, edge_train, edge_test, edge_val = sg.datasets.WN18().load()

EPOCHS = 100
EMBEDDING_DIMENSION = 200
NEGATIVE_SAMPLES = 10
LEARNING_RATE = 1E-3
PATIENCE = 10

def make_model():

    wn18_gen = KGTripleGenerator(
        dataset, batch_size=len(edge_train) // 100
    )
    wn18_complex = ComplEx(
        wn18_gen,
        embedding_dimension=EMBEDDING_DIMENSION,
        embeddings_regularizer=l2(1E-7)
    )
    wn18_inp, wn18_out = wn18_complex.in_out_tensors()

    wn18_model = Model(inputs=wn18_inp, outputs=wn18_out)
    wn18_model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss=BinaryCrossentropy(from_logits=True),
        metrics=[BinaryAccuracy(threshold=0.0)]
    )

    wn18_train_gen = wn18_gen.flow(
        edge_train, negative_samples=NEGATIVE_SAMPLES, shuffle=True
    )
    wn18_valid_gen = wn18_gen.flow(
        edge_val, negative_samples=NEGATIVE_SAMPLES
    )

    wn18_es = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    wn18_history = wn18_model.fit(
        wn18_train_gen, validation_data=wn18_valid_gen, epochs=EPOCHS, callbacks=[wn18_es]
    )

    sg.utils.plot_history(wn18_history)
    plt.show()

    return wn18_model


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

def predict_edge(model, edges):
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
    if len(positive_predictions) == 0:
        src, dst = edges[0][0], edges[0][1]
        print(f"No edge predicted between {src} and {dst} (real={k_graph.net.has_edge(src, dst)}).")
    else:
        for i, prediction in enumerate(positive_predictions):
            src, dst, pred = edges[i]
            print(f"Edge {src}-[{pred}]->{dst} predicted ({prediction}) (real={k_graph.net.has_edge(src, dst, pred)})")

if __name__ == "__main__" and False:
    from random import randrange, choice
    model = make_model()

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
        predict_edge(model, edges)
