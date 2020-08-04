import stellargraph as sg
import tensorflow.keras.backend as K
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
from dataset import make_dataset
from make_features import format2, format3

MODEL_NAME = "tranql-hinsage-model"

train_size = 0.8
test_size = 0.2

BATCH_SIZE = 10
EPOCHS = 20
NUM_SAMPLES = [8, 4]
HINSAGE_LAYER_SIZES = [32, 32]
LEARNING_RATE = 1E-2


def get_dataset(kg):
    return make_dataset(kg, format3)


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

    edgelist_train = list(edge_train[["source", "target"]].itertuples(index=False))
    edgelist_test = list(edge_test[["source", "target"]].itertuples(index=False))
    edgelist_val = list(edge_val[["source", "target"]].itertuples(index=False))

    labels_train = edge_train["label"]
    labels_test = edge_test["label"]
    labels_val = edge_val["label"]

    gen = HinSAGELinkGenerator(
        dataset, BATCH_SIZE, NUM_SAMPLES
    )
    train_gen = gen.flow(edgelist_train, labels_train, shuffle=True)
    test_gen = gen.flow(edgelist_test, labels_test)

    hinsage = HinSAGE(
        layer_sizes=HINSAGE_LAYER_SIZES,
        generator=gen,
        bias=True,
        dropout=0.0
    )

    x_inp, x_out = hinsage.in_out_tensors()
    prediction = link_regression(edge_embedding_method="concat")(x_out)

    def root_mean_square_error(s_true, s_pred):
        return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=LEARNING_RATE),
        loss=losses.mean_squared_error,
        metrics=[root_mean_square_error, metrics.mae]
    )

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        verbose=1,
        shuffle=False,
        use_multiprocessing=False,
        workers=4
    )

    sg.utils.plot_history(history)
    plt.show()
    return model


if __name__ == "__main__":
    from tranql_jupyter import KnowledgeGraph
    k_graph = KnowledgeGraph.mock("mock1.json")
    dataset = get_dataset(k_graph)
    model = make_model(dataset)