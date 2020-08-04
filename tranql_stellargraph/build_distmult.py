from stellargraph import utils
from stellargraph.mapper import KGTripleGenerator
from stellargraph.layer import DistMult
from tensorflow.keras import callbacks, optimizers, losses, metrics, regularizers, Model
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
from dataset import make_dataset

MODEL_NAME = "tranql-distmult-model"

train_size = 0.8
test_size = 0.2

EPOCHS = 300
EMBEDDING_DIMENSION = 100
NEGATIVE_SAMPLES = 2
LEARNING_RATE = 1E-3


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
        dataset, batch_size=len(edge_train)//10
    )

    distmult = DistMult(
        gen,
        embedding_dimension=EMBEDDING_DIMENSION,
        embeddings_regularizer=regularizers.l2(1E-7)
    )

    inp, out = distmult.in_out_tensors()

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(lr=LEARNING_RATE),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[metrics.BinaryAccuracy(threshold=0.0)]
    )

    train_gen = gen.flow(
        edge_train, negative_samples=NEGATIVE_SAMPLES, shuffle=True
    )
    valid_gen = gen.flow(
        edge_val, negative_samples=NEGATIVE_SAMPLES
    )

    es = callbacks.EarlyStopping(monitor="val_loss", patience=50)
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=[es],
        verbose=1
    )

    utils.plot_history(history)
    plt.show()

    return model


if __name__ == "__main__":
    from tranql_jupyter import KnowledgeGraph
    k_graph = KnowledgeGraph.mock("mock1.json")
    dataset = get_dataset(k_graph)
    model = make_model(dataset)
