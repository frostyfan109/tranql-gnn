import numpy as np
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator, DirectedGraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, DirectedGraphSAGE, link_classification
from tensorflow import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib.pyplot as plt
from tranql_jupyter import KnowledgeGraph
from tranql_stellargraph.dataset import make_dataset

k_graph = KnowledgeGraph.mock("mock1.json")
dataset = make_dataset(k_graph)

MODEL_NAME = "tranql-graphsage-model"


def make_model(G):
    p = 0.2
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=p, method="global"
    )
    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=p, method="global"
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
        dropout=0.5
    )

    X_in, X_out = graphsage.in_out_tensors()

    prediction = link_classification(
        output_dim=1,
        output_act="relu",
        edge_embedding_method="ip"
    )(X_out)

    model = Model(inputs=X_in, outputs=prediction)

    optimizer = Adam(lr=1E-3)
    model.compile(
        optimizer=optimizer,
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

    # model.save(MODEL_NAME)
    return model

def get_model(G):
    try:
        return keras.models.load_model(MODEL_NAME)
    except OSError:
        return make_model(G)


if __name__ == "__main__":
    from cora_stellargraph.build_graphsage import predict_random_edge, predict_random_real_edge

    model = get_model(dataset)

    for i in range(5): predict_random_edge(model, dataset)
    print()
    for i in range(5): predict_random_real_edge(model, dataset)
