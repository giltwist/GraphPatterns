from graph_pattern_common import GRAPH_REDUCTION_FACTOR, GRAPH_MODEL

import numpy as np
import pandas as pd
import time
from math import ceil

import multiprocessing

from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter



import dill

# == THIS SECTION COMES MORE OR LESS VERBATIM FROM A TUTORIAL ==
# StellarGraph metapath2vec tutorial - https://stellargraph.readthedocs.io/en/latest/demos/link-prediction/metapath2vec-link-prediction.html

def split_train_test(graph):

    # Config options
    dimensions = 128
    num_walks = 1
    walk_length = 10
    context_window_size = 10
    num_iter = 1
    workers = multiprocessing.cpu_count()

    # Review of same product OR of products that are no more than 3 category hops away
    user_metapaths = [
        ["user", "product", "user"],
        ["user", "product", "category", "product", "user"],
        ["user", "product", "category", "category", "product", "user"],
        ["user", "product", "category", "category", "category", "product", "user"],
        ["product","category","product"],
        ["category","category"]
    ]

    # BEGIN HELPER FUNCTIONS

    def metapath2vec_embedding(graph, name):
        rw = UniformRandomMetaPathWalk(graph)
        walks = rw.run(
            graph.nodes(), n=num_walks, length=walk_length, metapaths=user_metapaths
        )
        print(f"Number of random walks for '{name}': {len(walks)}")

        model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=context_window_size,
            min_count=0,
            sg=1,
            workers=workers,
            epochs=num_iter,
        )

        def get_embedding(u):
            return model.wv[u]

        return get_embedding


    # 1. link embeddings
    def link_examples_to_features(link_examples, transform_node, binary_operator):
        return [
            binary_operator(transform_node(src), transform_node(dst))
            for src, dst in link_examples
        ]


    # 2. training classifier
    def train_link_prediction_model(
        link_examples, link_labels, get_embedding, binary_operator
    ):
        clf = link_prediction_classifier()
        link_features = link_examples_to_features(
            link_examples, get_embedding, binary_operator
        )
        clf.fit(link_features, link_labels)
        return clf


    def link_prediction_classifier(max_iter=2000):
        lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
        return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


    # 3. and 4. evaluate classifier
    def evaluate_link_prediction_model(
        clf, link_examples_test, link_labels_test, get_embedding, binary_operator
    ):
        link_features_test = link_examples_to_features(
            link_examples_test, get_embedding, binary_operator
        )
        score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
        return score


    def evaluate_roc_auc(clf, link_features, link_labels):
        predicted = clf.predict_proba(link_features)
        
        # check which class corresponds to positive links
        positive_column = list(clf.classes_).index(1)
        return roc_auc_score(link_labels, predicted[:, positive_column])

    def operator_l1(u, v):
        return np.abs(u - v)


    def operator_l2(u, v):
        return (u - v) ** 2


    def run_link_prediction(binary_operator):
        clf = train_link_prediction_model(
            examples_train, labels_train, embedding_train, binary_operator
        )
        score = evaluate_link_prediction_model(
            clf,
            examples_model_selection,
            labels_model_selection,
            embedding_train,
            binary_operator,
        )

        return {
            "classifier": clf,
            "binary_operator": binary_operator,
            "embedding": embedding_train,
            "score": score,
        }
    # END HELPER FUNCTIONS

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Sampling reviews and reviews-complement (est. ~{ceil(360/GRAPH_REDUCTION_FACTOR)} seconds)"))

    edge_splitter_test = EdgeSplitter(graph)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", edge_label="review"
    )

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tSampling time: {int(end-start)}s"))


    start = time.time()
    print("\033[91m{}\033[00m".format(f"Building training set (est. ~{ceil(600/GRAPH_REDUCTION_FACTOR)} seconds)"))

    edge_splitter_train = EdgeSplitter(graph_test, graph)
    
    graph_train, examples, labels = edge_splitter_train.train_test_split(
    p=0.1, method="global", edge_label="review"
    )
    
    (
    examples_train,
    examples_model_selection,
    labels_train,
    labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tBuilding time: {int(end-start)}s"))

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Embedding training set (est. ~{ceil(1800/GRAPH_REDUCTION_FACTOR)} seconds)"))
    embedding_train = metapath2vec_embedding(graph_train, "Train Graph")
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tEmbedding time: {int(end-start)}s"))


    binary_operators = [operator_l1, operator_l2]

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Comparing binary operators (est. ~{ceil(600/GRAPH_REDUCTION_FACTOR)} seconds)"))
    results = [run_link_prediction(op) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])
    
    print(f"Best result from '{best_result['binary_operator'].__name__}'")

    op_table = pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")
    print(op_table)

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tComparison time: {int(end-start)}s"))

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Embedding testing set (est. ~{ceil(1800/GRAPH_REDUCTION_FACTOR)} seconds)"))
    embedding_test = metapath2vec_embedding(graph_test, "Test Graph")
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tEmbedding time: {int(end-start)}s"))

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Evaluating testing set (est. ~X minutes)"))
    test_score = evaluate_link_prediction_model(
    best_result["classifier"],
    examples_test,
    labels_test,
    embedding_test,
    best_result["binary_operator"],
    )
    print(
    f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
    )
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tEvaluation time: {int(end-start)}s"))

    with open(GRAPH_MODEL, 'wb') as file:
        dill.dump(best_result,file)

    return best_result




# == END TUTORIAL SECTION ==