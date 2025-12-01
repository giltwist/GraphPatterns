from graph_pattern_common import parse_metadata, get_title
from graph_pattern_common import GRAPH_REDUCTION_FACTOR, GRAPH_META, GRAPH_MODEL, GRAPH_CATEGORIES, GRAPH_VECTORIZER, GRAPH_REVIEWS
from graph_NN import split_train_test


import os
import time
import networkx as nx
import numpy as np
import pandas as pd

from math import ceil, log2

import matplotlib.pyplot as plt

from stellargraph import StellarGraph

from keras.saving import load_model

import dill

from random import sample

from sklearn.feature_extraction.text import TfidfVectorizer

def generate_vectorizer():

    def pipe_tokenize(text):
        return text.split('|')


    if os.path.exists(GRAPH_META):
        corpus=[]
        for i, e in enumerate(parse_metadata(GRAPH_META)):
            
            if i%GRAPH_REDUCTION_FACTOR==0:            
                if 'categories' in e:
                    item_categories=""
                    for j, c in enumerate(e['categories']):
                        #remove leading pipe
                        if j==0:
                            item_categories+=c[1:]
                        else:
                            item_categories+=c
                    corpus.append(item_categories)
                     
        vectorizer = TfidfVectorizer(tokenizer=pipe_tokenize,max_df=0.7,max_features=100)
        vectorizer=vectorizer.fit(corpus)
        with open(GRAPH_VECTORIZER, 'wb') as file:
            dill.dump(vectorizer, file)

        return vectorizer
    else:
        print("Amazon metadata not found.  Download it from  https://snap.stanford.edu/data/amazon-meta.html")



# NetworkX has nicer building and storing functions for graphs than StellarGraph
def generate_graph():

    if os.path.exists(GRAPH_META):

        graph = nx.Graph()
        reviews_array = []

        #all_users=[]

        # can use iterator i for limiting loop
        # otherwise use entry e for data access
        for i, e in enumerate(parse_metadata(GRAPH_META)):
            
            #Useful for debugging
            #print(str(i) + "|" + simplejson.dumps(e, indent=4) + "\n\n")

            if i%GRAPH_REDUCTION_FACTOR==0:            
                # Ignore products with no categories or no reviews
                if 'categories' in e and 'reviews' in e:
                    asin = e['ASIN']
                    graph.add_node(asin,type='product')
                    item_categories=""
                    for j, c in enumerate(e['categories']):
                        #remove leading pipe
                        if j==0:
                            item_categories+=c[1:]
                        else:
                            item_categories+=c
                    graph.nodes[asin]['feature']=np.array(vectorizer.transform([item_categories]).toarray().flatten())
                    for r in e['reviews']:
                        # Nodes must have at least one feature, but we have no data on users so all get the same
                        graph.add_node(r['customer'],type='user',feature=[1])
                        graph.add_edge(r['customer'],asin)
                        reviews_array.append({'user':r['customer'],'product':asin,'rating':r['rating']})

        review_df = pd.DataFrame(reviews_array)
        with open(GRAPH_CATEGORIES, 'wb') as file:
            dill.dump(graph, file)
        with open(GRAPH_REVIEWS, 'wb') as file:
            dill.dump(review_df, file)
        return graph, review_df
    else:
        print("Amazon metadata not found.  Download it from  https://snap.stanford.edu/data/amazon-meta.html")

#Must be NetworkX Graph
def get_reviews(user, graph):
    neighbors=dict(graph[user])
    dub_tab="\n\t\t"
    reviews=[]
    for neighbor in neighbors:
        reviews.append(f"{'Likes: ' if neighbors[neighbor]['weight']>=3 else 'Dislikes: '}{get_title(neighbor,GRAPH_META)}\n\t\t{dub_tab.join(get_types(neighbor,graph))}")
    return reviews

#Must be NetworkX Graph
def get_types(node, graph):
    neighbors = dict(graph[node])
    types = []
    for neighbor in neighbors:
        if neighbors[neighbor]['type']=="product_type":
            types.append(neighbor)
    return types
        

              
# NOTE: Time estimates were derived on an AMD 3770X CPU with 16GB memory
if __name__ == "__main__":

    

    # Loading is much faster than generating, particular with low GRAPH_REDUCTION_FACTOR


    nx_graph = nx.Graph()
    vectorizer=TfidfVectorizer()
    reviews=pd.DataFrame()

    trained=None
    

    # Try to load the vectorizer
    if os.path.exists(GRAPH_VECTORIZER):
        print("\033[91m{}\033[00m".format(f"Loading existing vectorizer (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        with open(GRAPH_VECTORIZER, 'rb') as file:
            vectorizer = dill.load(file)
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))
    
    else:
        print("\033[91m{}\033[00m".format(f"Generating new vectorizer (est. ~60 seconds)"))
        start = time.time()
        vectorizer = generate_vectorizer()
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tGeneration time: {int(end-start)}s"))

    # Try to load the graph/reviews
    if os.path.exists(GRAPH_CATEGORIES) and os.path.exists(GRAPH_REVIEWS):
        print("\033[91m{}\033[00m".format(f"Loading existing category graph (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        with open(GRAPH_CATEGORIES, 'rb') as file:
            nx_graph = dill.load(file)
        #Restore node types not saved in edgelist
        for node in nx_graph.nodes:
            if len(node)==10 and not node.startswith('A'):
                nx_graph.nodes[node]['type']='product'
            else:
                nx_graph.nodes[node]['type']='user'
        
        with open(GRAPH_REVIEWS, 'rb') as file:
            reviews = dill.load(file)

        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))
    else:
        # Generate everything
        print("\033[91m{}\033[00m".format(f"Generating new category graph (est. ~65 seconds)"))
        start = time.time()
        nx_graph, reviews = generate_graph()
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tGeneration time: {int(end-start)}s"))


    # Send NetworkX graph to StellarGraph format
    print("\033[91m{}\033[00m".format(f"Activating StellarGraph Library (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
    start = time.time()
    stellar_graph = StellarGraph.from_networkx(nx_graph,node_type_attr='type',edge_type_default='review',node_features='feature')
    print(stellar_graph.info())
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tActivation time: {int(end-start)}s"))

    # Try to load complete model
    if os.path.exists(GRAPH_MODEL):
        print("\033[91m{}\033[00m".format(f"Loading existing link prediction model (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        trained = load_model(GRAPH_MODEL)
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))

    else:
        # Do actual training of GNN using Metapath2Vec
        trained = split_train_test(stellar_graph, reviews)
    