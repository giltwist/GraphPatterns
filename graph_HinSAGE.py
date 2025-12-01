from graph_pattern_common import GRAPH_REDUCTION_FACTOR, GRAPH_HINSAGE_MODEL, GRAPH_HINSAGE_GENERATOR

import time

import json
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stellargraph import StellarGraph
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from keras import Model, optimizers, losses, metrics

import multiprocessing

import dill

from typing import Tuple

# == THIS SECTION COMES MORE OR LESS VERBATIM FROM A TUTORIAL ==
# https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/hinsage-link-prediction.html

# Split, train, and test HinSAGE Model
# This model is good at predicting ratings of hypothetical edges
# BUT it does not predict the likelihood of the existence of the edge
def stt_HinSAGE(G: StellarGraph, edges_with_ratings: pd.DataFrame) -> Tuple[Model,HinSAGELinkGenerator]:

    # BEGIN CONFIG SECTION
    batch_size = 200
    epochs = 20
    # Use 70% of edges for training, the rest for testing:
    train_size = 0.7
    test_size = 0.3
    # layers/iterations of HinSAGE model
    num_samples = [8, 4]


    hinsage_layer_sizes = [32, 32]
    assert len(hinsage_layer_sizes) == len(num_samples)


    num_workers = multiprocessing.cpu_count()
    # END CONFIG

    #BEGIN HELPER FUNCTION SECTION

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Splitting data set for train/test (est. ~X seconds)"))

    edges_train, edges_test = model_selection.train_test_split(
        edges_with_ratings, train_size=train_size, test_size=test_size
    )

    edgelist_train = list(edges_train[["user", "product"]].itertuples(index=False))
    edgelist_test = list(edges_test[["user", "product"]].itertuples(index=False))


    labels_train = edges_train["rating"]
    labels_test = edges_test["rating"]

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tSplitting time: {int(end-start)}s"))


    start = time.time()
    print("\033[91m{}\033[00m".format(f"Preparing HinSage Model (est. ~X seconds)"))

    generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["user", "product"]
)
    train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
    test_gen = generator.flow(edgelist_test, labels_test)

    generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples))

    hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0)

    x_inp, x_out = hinsage.in_out_tensors()
    score_prediction = link_regression(edge_embedding_method="concat")(x_out)


    model = Model(inputs=x_inp, outputs=score_prediction)
    model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=losses.mean_squared_error,
    metrics=[metrics.RootMeanSquaredError(),metrics.mae],
    )

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tPreparation time: {int(end-start)}s"))



    start = time.time()
    print("\033[91m{}\033[00m".format(f"Beginning model test/training (est. ~X seconds)"))

    test_metrics = model.evaluate(
    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
    )

    print("Untrained model's Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tPreparation time: {int(end-start)}s"))


    history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers,
    )

    test_metrics = model.evaluate(
    test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
)

    print("Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    y_true = labels_test
    # Predict the rankings using the model:
    y_pred = model.predict(test_gen)
    # Mean baseline rankings = mean movie ranking:
    y_pred_baseline = np.full_like(y_pred, np.mean(y_true))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
    mae = mean_absolute_error(y_true, y_pred_baseline)
    print("Mean Baseline Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("\nModel Test set metrics:")
    print("\troot_mean_square_error = ", rmse)
    print("\tmean_absolute_error = ", mae)   

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tTraining/Testing time: {int(end-start)}s"))
    
    model.save(GRAPH_HINSAGE_MODEL)
    with open(GRAPH_HINSAGE_GENERATOR, 'wb') as file:
        dill.dump(generator, file)

    return model, generator




# == END TUTORIAL SECTION ==