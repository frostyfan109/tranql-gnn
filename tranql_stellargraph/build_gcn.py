import numpy as np
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
from tensorflow import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
from dataset import make_dataset
from make_features import format3


def get_dataset(kg):
    return make_dataset(kg, format3)

def make_model(G):
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.2, method="global"
    )
    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.2, method="global"
    )

    epochs = 50

    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=train_gen,
        dropout=0.3
    )
    x_inp, x_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
    prediction = Reshape((-1, ))(prediction)

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=Adam(lr=0.01),
        loss=binary_crossentropy,
        metrics=["binary_accuracy"]
    )
    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_flow,
        epochs=epochs,
        validation_data=test_flow,
        verbose=2,
        shuffle=False
    )

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)

    # print("\nTrain Set Metrics of the trained model:")
    # for name, val in zip(model.metrics_names, train_metrics):
    #     print("\t{}: {:0.4f}".format(name, val))
    #
    # print("\nTest Set Metrics of the trained model:")
    # for name, val in zip(model.metrics_names, test_metrics):
    #     print("\t{}: {:0.4f}".format(name, val))

    sg.utils.plot_history(history)
    plt.show()

    return model


def main2():
    from tranql_jupyter import KnowledgeGraph
    k_graph = KnowledgeGraph.mock("mock1.json")
    dataset = get_dataset(k_graph)
    model = make_model(dataset)


def main():
    from random import randrange
    from tranql_jupyter import KnowledgeGraph

    k_graph = KnowledgeGraph.mock("mock1.json")
    dataset = get_dataset(k_graph)
    model = make_model(dataset)

    net = dataset.to_networkx()
    nodes = list(net.nodes)
    edges = list(net.edges)

    edge_ids = [
        [nodes.pop(randrange(len(nodes))) for i in range(2)] for n in range(5)
    ]
    edge_ids += [
        edges.pop(randrange(len(edges)))[:2] for n in range(5)
    ]
    # Within mock1.json, the only edge from CHEMBL:520007 is to the gene HGNC:9591 (Prostaglandin D2 Receptor/PTGDR)
    # The model should strongly predict an edge from CHEMBL:520007 to HGNC:9594    (Prostaglandin E Receptor 2/PTGER2)
    # This is interesting because PTGER2 is an important paralog of PTGDR
    # See: section "GeneCards Summary for PTGDR Gene" from https://www.genecards.org/cgi-bin/carddisp.pl?gene=PTGDR
    edge_ids.append(["HGNC:10610", "MONDO:0010940"])
    edge_ids = np.array(edge_ids)
    edge_labels = np.array([
        net.has_edge(n0, n1) for (n0, n1) in edge_ids
    ])

    gen = FullBatchLinkGenerator(dataset, method="gcn")
    flow = gen.flow(edge_ids, edge_labels)

    predictions = model.predict(flow)
    for i, prediction in enumerate(predictions[0]):
        prediction = prediction[0]
        n0, n1 = edge_ids[i]
        real = net.has_edge(n0, n1)
        print(f"{n0}->{n1}. Prediction: {prediction}. Real: {real}.")


if __name__ == "__main__":
    # main()
    main2()