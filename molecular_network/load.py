import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from spektral.layers import GraphAttention, GlobalAttentionPool
from spektral.utils import load_sdf, nx_to_numpy, label_to_one_hot
from spektral.chem import sdf_to_nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sdf_loaded = load_sdf('Trainingset_Delaunay-CHNO5000selected.sdf', amount=None)
sdf_nx = sdf_to_nx(sdf_loaded, keep_hydrogen=True)

sdf_adj, sdf_node, _ = nx_to_numpy(sdf_nx,
                                   nf_keys=['atomic_num'],
                                   ef_keys=['type'])


uniq_node = np.unique([v for x in sdf_node for v in np.unique(x)]) 
node = [label_to_one_hot(x, uniq_node) for x in sdf_node]

label_pd = pd.read_csv('Trainingset_Delaunay-CHNO5000selected_label.csv')
label = label_pd.to_numpy() # Two classes: low (logP <= 3) and high (logP > 3)

N = node[0].shape[-2]          # Number of nodes in the graphs
F = node[0].shape[-1]          # Original feature dimensionality (4 features)
n_classes = label.shape[-1]    # Number of classes (2)
l2_reg = 5e-4                  # Regularization rate for l2
learning_rate = 1e-4           # Learning rate for Adam
epochs = 50                    # Number of training epochs
batch_size = 32                # Batch size
es_patience = 5                # Patience for early stopping
es_delta = 0.001               # Min Delta for early stopping

def make_model():
    A_train, A_test, \
    X_train, X_test, \
    y_train, y_test = train_test_split(sdf_adj, node, label, test_size=0.1)

    X_in = Input(shape=(N, F))
    A_in = Input((N, N))

    gc1 = GraphAttention(32,
                         activation="relu",
                         kernel_regularizer=l2(l2_reg))([X_in, A_in])
    gc2 = GraphAttention(64,
                         activation="relu",
                         kernel_regularizer=l2(l2_reg))([gc1, A_in])
    pool = GlobalAttentionPool(128)(gc2)
    output = Dense(n_classes, activation="softmax")(pool)

    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    history = model.fit([X_train, A_train],
              y_train,
              batch_size=batch_size,
              validation_split=0.1,
              epochs=epochs,
              callbacks=[
                  EarlyStopping(monitor="val_loss",
                                min_delta=es_delta,
                                patience=es_patience,
                                verbose=1,
                                restore_best_weights=True)
              ])
