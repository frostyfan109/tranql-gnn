import numpy as np
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tranql_jupyter import KnowledgeGraph
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from tranql_stellargraph.dataset import k_graph, dataset

MODEL_NAME = "tranql-hinsage-model"

batch_size = 20
epochs = 20
train_size = 0.7
test_size = 0.3

edges_with_weights = pd.DataFrame([
    {
        "source_id": e["source_id"],
        "target_id": e["target_id"],
        "weight": e["weight"]
    } for e in k_graph.build_knowledge_graph()["edges"]
])

edges_train, edges_test = model_selection.train_test_split(
    edges_with_weights, train_size=train_size, test_size=test_size
)
edgelist_train = list(edges_train[["source_id", "target_id"]].itertuples(index=False))
edgelist_test = list(edges_test[["source_id", "target_id"]].itertuples(index=False))

labels_train = edges_train["weight"]
labels_test = edges_test["weight"]


def make_model(G):
    num_samples = [8, 4]
    generator = HinSAGELinkGenerator(
        dataset, batch_size, num_samples
    )
    train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
    test_gen = generator.flow(edgelist_test, labels_test)

    layer_sizes = [32, 32]
    hinsage = HinSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=True,
        dropout=0
    )
    x_inp, x_out = hinsage.in_out_tensors()

    score_prediction = link_regression(edge_embedding_method="concat")(x_out)

    def root_mean_square_error(s_true, s_pred):
        K = keras.backend
        return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))

    model = Model(inputs=x_inp, outputs=score_prediction)
    model.compile(
        optimizer=Adam(lr=1E-2),
        loss=keras.losses.mean_squared_error,
        metrics=[root_mean_square_error, keras.metrics.mae]
    )
    model.summary()

    num_workers = 4
    test_metrics = model.evaluate(
        test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
    )
    print("Untrained model's Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        verbose=1,
        shuffle=False,
        use_multiprocessing=False,
        workers=num_workers
    )

    sg.utils.plot_history(history)
    plt.show()

    test_metrics = model.evaluate(
        test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
    )

    print("Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))