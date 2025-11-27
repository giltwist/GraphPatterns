
import os
import time
from graph_pattern_common import parse_metadata

import matplotlib.pyplot as plt
from math import isclose, ceil
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from sklearn.model_selection import train_test_split

from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# == GLOBAL CONFIG ==

# From https://snap.stanford.edu/data/amazon-meta.html
GRAPH_META = "./data/amazon-meta.txt"

# Graph we are building
GRAPH_CATEGORIES = "./data/amazon-categories.txt"

# NOTE: Accepts one in GRAPH_REDUCTION_FACTOR products from the original dataset to accommodate system memory
# Uncomment only one line
# For a quick run, a factor of 10 is recommended
GRAPH_REDUCTION_FACTOR = 10
# With 16 GB of memory, factor of 2 is recommended
#GRAPH_REDUCTION_FACTOR = 2
# If you have 32 or 64GB of memory, a factor of 1 is likely achievable
#GRAPH_REDUCTION_FACTOR = 1

# == END CONFIG == 

# NetworkX has nicer building and storing functions for graphs than StellarGraph
def generate_graph():

    if os.path.exists(GRAPH_META):

        graph = nx.Graph()

        all_users=[]


        # can use iterator i for limiting loop
        # otherwise use entry e for data access
        for i, e in enumerate(parse_metadata(GRAPH_META)):
            
            #Useful for debugging
            #print(str(i) + "|" + simplejson.dumps(e, indent=4) + "\n\n")


            if i%GRAPH_REDUCTION_FACTOR==0:
            
                asin = e['ASIN']
                graph.add_node(asin,type='product')
                if 'reviews' in e:
                    for r in e['reviews']:
                        graph.add_node(r['customer'],type='user')
                        
                        graph.add_edge(r['customer'],asin,weight = r['rating'], type='review')
                        all_users.append(r['customer'])
                if 'categories' in e:
                    for c in e['categories']:
                        #print(c)
                        info = c.split("|")[1:]
                        for i in range(len(info)-1):
                            graph.add_node(info[i],type='category')
                            graph.add_node(info[i+1],type='category')
                            graph.add_edge(info[i],info[i+1],weight = 0, type='category_hierarchy')
                        graph.add_edge(info[len(info)-1],asin,weight = 0, type='product_type')
                #TODO: Add product-to-product edges with the similar-to information.
        
        #deduplicate users
        all_users=set(all_users)
        print(f"Total users: {len(all_users)}")

        #Remove any leaf nodes
        print("Pruning singleton reviewers.")
        leaves = [node for node,degree in dict(graph.degree()).items() if degree == 1]
        multi_users=all_users-set(leaves)
        print(f"Reviewers with multiple reviews: {len(multi_users)}.")
        graph.remove_nodes_from(leaves)


        # NOTE: Tried GraphML first because it stored node types as well, but was 1.1GB
        # NOTE: Switched to edge list based on research that suggested it was more memory efficient, reduced size to 300MB
        nx.write_edgelist(graph,GRAPH_CATEGORIES,delimiter='|', data=['weight','type'])
        return graph
    else:
        print("Amazon metadata not found.  Download it from  https://snap.stanford.edu/data/amazon-meta.html")


def visualize_graph(graph):
    # Only for small numbers of nodes
    plt.figure(figsize=(60, 60)) 
    if (graph.number_of_nodes()<1000 and graph.number_of_edges()<1000):
        
        nx.draw(graph, with_labels=True, node_size=50)
    else:
        print("Graph too large to visualize")

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
            "score": score,
        }
    # END HELPER FUNCTIONS

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Sampling reviews and reviews-complement (est. ~{ceil(6/GRAPH_REDUCTION_FACTOR)} minutes)"))

    edge_splitter_test = EdgeSplitter(graph)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", edge_label="review"
    )

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tSampling time: {int(end-start)}s"))


    start = time.time()
    print("\033[91m{}\033[00m".format(f"Building training set (est. ~{ceil(10/GRAPH_REDUCTION_FACTOR)} minutes)"))

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
    print("\033[91m{}\033[00m".format(f"Embedding training set (est. ~{ceil(30/GRAPH_REDUCTION_FACTOR)} minutes)"))
    embedding_train = metapath2vec_embedding(graph_train, "Train Graph")
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tEmbedding time: {int(end-start)}s"))


    binary_operators = [operator_l1, operator_l2]

    results = [run_link_prediction(op) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])

    print(f"Best result from '{best_result['binary_operator'].__name__}'")

    op_table = pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")
    print(op_table)

    start = time.time()
    print("\033[91m{}\033[00m".format(f"Embedding testing set (est. ~{ceil(30/GRAPH_REDUCTION_FACTOR)} minutes)"))
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

    return best_result




# == END TUTORIAL SECTION ==


def estimate_progress(task,duration):
    pass

# NOTE: Time estimates were derived on an AMD 3770X CPU with 16GB memory
if __name__ == "__main__":

    # Loading is much faster than generating, particular with high GRAPH_REDUCTION_FACTOR
    if os.path.exists(GRAPH_CATEGORIES):
        print("\033[91m{}\033[00m".format(f"Loading existing category graph (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        nx_graph = nx.read_edgelist(GRAPH_CATEGORIES,delimiter='|',nodetype=str, data=(('weight',int),('type',str)),comments=None)
        #Restore node types not saved in edgelist
        for node in nx_graph.nodes:
            if '[' in node:
                nx_graph.nodes[node]['type']='category'
            elif len(node)==10 and not node.startswith('A'):
                nx_graph.nodes[node]['type']='product'
            else:
                nx_graph.nodes[node]['type']='user'
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))
    else:
        print("\033[91m{}\033[00m".format(f"Generating new category graph (est. ~90 seconds)"))
        start = time.time()
        nx_graph = generate_graph()
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tGeneration time: {int(end-start)}s"))


    # Send NetworkX graph to StellarGraph format

    print("\033[91m{}\033[00m".format(f"Activating StellarGraph Library (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
    start = time.time()
    stellar_graph = StellarGraph.from_networkx(nx_graph,node_type_attr='type',edge_type_attr='type', edge_weight_attr='weight')
    print(stellar_graph.info())
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tActivation time: {int(end-start)}s"))

    split_train_test(stellar_graph)


        
        
        

    #visualize_graph(graph)

            
