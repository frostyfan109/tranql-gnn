import numpy as np
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
from tensorflow import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib.pyplot as plt

def make_model():
    G, _ = sg.datasets.Cora().load(subject_as_feature=True)

    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )
    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    epochs = 50

    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    # Is this supposed to be train_gen.flow??
    test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=train_gen,
        dropout=0.3
    )

    X_in, X_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="relu", method="ip")(X_out)
    # Reshape predictions from (X, 1) to (X,)
    prediction = Reshape((-1, ))(prediction)

    model = Model(inputs=X_in, outputs=prediction)

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
    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    sg.utils.plot_history(history)
    plt.show()

    return model

if __name__ == "__main__":
    model = make_model()