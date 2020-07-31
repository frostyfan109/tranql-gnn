import numpy as np
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib.pyplot as plt
from random import choice

MODEL_NAME = "graphsage-model"


def make_model(G):
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )
    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    batch_size = 20
    epochs = 20
    num_samples = [20, 10]

    train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

    test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

    layer_sizes = [20, 20]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=train_gen,
        bias=True,
        dropout=0.3
    )

    X_in, X_out = graphsage.in_out_tensors()

    prediction = link_classification(
        output_dim=1,
        output_act="relu",
        edge_embedding_method="ip"
    )(X_out)

    model = Model(inputs=X_in, outputs=prediction)

    model.compile(
        optimizer=Adam(lr=1E-3),
        loss=binary_crossentropy,
        metrics=["acc"]
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
        verbose=2
    )

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)
    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    sg.utils.plot_history(history)
    plt.show()

    model.save(MODEL_NAME)
    return model

def get_model(G):
    try:
        return keras.models.load_model(MODEL_NAME)
    except OSError:
        return make_model(G)

def predict_random_real_edge(model, G):
    net = G.to_networkx()
    edges = list(net.edges())

    n1, n2 = choice(edges)
    predict_random_edge(model, G, n1=n1, n2=n2)


def predict_random_edge(model, G, n1=None, n2=None):
    net = G.to_networkx()
    nodes = list(net.nodes())

    if n1 is None or n2 is None:
        n1 = choice(nodes)
        nodes.remove(n1)
        n2 = choice(nodes)
        nodes.remove(n2)

    edge_ids = np.array([
        [n1, n2]
    ])
    edge_labels = np.array([
        net.has_edge(n1, n2)
    ])

    batch_size = 20
    num_samples = [20, 10]
    gen = GraphSAGELinkGenerator(G, batch_size, num_samples)
    flow = gen.flow(edge_ids, edge_labels)
    predictions = model.predict(flow)
    prediction = predictions[0, 0]

    print(f"Prediction: {round(prediction * 100, 1)}% confident that an edge could exist between {n1} and {n2}. Edge exists: {edge_labels[0]}")

    return prediction, edge_ids[0], edge_labels[0]

if __name__ == "__main__":
    G, _ = sg.datasets.Cora().load(subject_as_feature=True)
    model = get_model(G)

    print("Random nodes:")
    for i in range(5): predict_random_edge(model, G)
    print("Real edges:")
    for i in range(5): predict_random_real_edge(model, G)