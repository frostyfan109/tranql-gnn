import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from spektral.layers import GraphConv
from sklearn.model_selection import train_test_split
from tranql_jupyter import KnowledgeGraph
from make_data import make_data, make_masks


def make_model():
    k_graphs = (KnowledgeGraph.mock1(), KnowledgeGraph.mock2(), KnowledgeGraph.mock("mock3.json"))

    A, (type_names, X), _, (reasoner_names, y) = make_data(k_graphs)

    N = X[0].shape[-2] # Number of nodes in the graphs
    F = X[0].shape[-1] # Original feature dimensionality
    n_classes = y.shape[-1] # Number of classes

    X_in = Input(shape=(N, F))
    A_in = Input((N, N))

    X_1 = GraphConv(16, "relu")([X_in, A_in])


def make_model():
    k_graphs = (KnowledgeGraph.mock("mock5.json"),)

    A, (type_names, X), _, (reasoner_names, y) = make_data(k_graphs)
    A, X, y = A[0], X[0], y[0]

    N = X.shape[-2]
    F = X.shape[-1]
    n_classes = y.shape[-1]
    batch_size, shuffle, epochs = N, False, 1

    train_mask, val_mask, test_mask = make_masks(N, val=0.1, test=0.1)

    X_in = Input(shape=(F, ))
    A_in = Input(shape=(N, ))

    X_1 = GraphConv(16, "relu")([X_in, A_in])
    X_1 = Dropout(0.5)(X_1)
    X_2 = GraphConv(n_classes, "softmax")([X_1, A_in])

    model = Model(inputs=[X_in, A_in], outputs=X_2)

    A = GraphConv.preprocess(A).astype("f4")

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  weighted_metrics=["acc"])
    model.summary()

    A = A.astype("f4")
    validation_data = ([X, A], y, val_mask)

    """
    A_train, A_test, \
    X_train, X_test, \
    y_train, y_test = train_test_split(A, X, y, test_size=0.1) # ?? In[0]: [165,205], In[1]: [165,16]
    """

    model.fit([X, A],
              y,
              sample_weight=train_mask,
              validation_data=validation_data,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=shuffle)

    eval_results = model.evaluate([X, A],
                                  y,
                                  sample_weight=test_mask,
                                  batch_size=batch_size)

    print("Done.\n"
          "Test loss: {}\n"
          "Test accuracy: {}".format(*eval_results))

    return model



def get_model():
    """
    Returns either a saved instance of the model or builds and trains the model on the spot
    """
    try:
        model = keras.models.load_model("tranql-model")
    except OSError:
        model = make_model()

    return model