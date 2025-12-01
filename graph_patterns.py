import os
import time
from math import ceil
from random import sample

import dill
import networkx as nx
import numpy as np
import pandas as pd
from keras.saving import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from stellargraph import StellarGraph

from typing import Tuple, List

from graph_HinSAGE import stt_HinSAGE
from graph_Metapath2Vec import stt_Metapath2Vec

from graph_pattern_common import (GRAPH_CATEGORIES, GRAPH_HINSAGE_GENERATOR,
                                  GRAPH_META, GRAPH_HINSAGE_MODEL,
                                  GRAPH_REDUCTION_FACTOR, GRAPH_REVIEWSDF,
                                  GRAPH_VECTORIZER, GRAPH_METAPATH2VEC_MODEL, get_categories, get_title,
                                  parse_metadata)


def generate_vectorizer() -> TfidfVectorizer:

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
def generate_nxgraph_reviewsdf() -> Tuple[nx.Graph, pd.DataFrame]:

    if os.path.exists(GRAPH_META):

        graph = nx.Graph()
        reviews_array = []

        # can use iterator i for limiting loop
        # otherwise use entry e for data access
        for i, e in enumerate(parse_metadata(GRAPH_META)):
            

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
        with open(GRAPH_REVIEWSDF, 'wb') as file:
            dill.dump(review_df, file)
        return graph, review_df
    else:
        print("Amazon metadata not found.  Download it from  https://snap.stanford.edu/data/amazon-meta.html")

#Must be NetworkX Graph
def get_reviews(user: str, graph: nx.Graph, review_df: pd.DataFrame) -> List[str]:
    neighbors=dict(graph[user])
    dub_tab="\n\t\t"
    reviews=[]
    for neighbor in neighbors:
        rating = list(review_df[(review_df['user']==user) & (review_df['product']==neighbor)]['rating'])[0]
        reviews.append(f"{'Likes: ' if rating>=3 else 'Dislikes: '}{get_title(neighbor,GRAPH_META)}\n\t\t{dub_tab.join(get_categories(neighbor,GRAPH_META))}")
    return reviews


              
# NOTE: Time estimates were derived on an AMD 3770X CPU with 16GB memory
if __name__ == "__main__":

    # Init variables to set scope
    nx_graph = None
    stellar_graph = None
    vectorizer = None
    reviews = None

    trained_hinsage_model = None
    trained_hinsage_generator = None
    
    trained_metapath2vec_clf = None
    trained_metapath2vec_op = None
    trained_metapath2vec_embedding = None

    
    # Load or generate the TF-IDF vectorizer for category texts
    # This is primarily used by HinSAGE
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

    # Load or generate a NetworkX Graph reviews DataFrame
    # These are used by both models
    if os.path.exists(GRAPH_CATEGORIES) and os.path.exists(GRAPH_REVIEWSDF):
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
        
        with open(GRAPH_REVIEWSDF, 'rb') as file:
            reviews = dill.load(file)

        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))
    else:
        print("\033[91m{}\033[00m".format(f"Generating new category graph (est. ~65 seconds)"))
        start = time.time()
        nx_graph, reviews = generate_nxgraph_reviewsdf()
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tGeneration time: {int(end-start)}s"))


    # Send NetworkX graph to StellarGraph format
    print("\033[91m{}\033[00m".format(f"Activating StellarGraph Library (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
    start = time.time()
    stellar_graph = StellarGraph.from_networkx(nx_graph,node_type_attr='type',edge_type_default='review',node_features='feature')
    print(stellar_graph.info())
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tActivation time: {int(end-start)}s"))

    # Load or generate Metapath2Vec Model
    # This model is good at predicting the likelihood of the existence of the edge
    # BUT it does not predict the ratings of hypothetical edges
    if os.path.exists(GRAPH_METAPATH2VEC_MODEL):
        print("\033[91m{}\033[00m".format(f"Loading existing link prediction model (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        
        with open(GRAPH_METAPATH2VEC_MODEL, 'rb') as file:
            trained = dill.load(file)
        
        trained_metapath2vec_clf = trained['classifier']
        trained_metapath2vec_op = trained['binary_operator']
        trained_metapath2vec_embedding = trained['embedding']

        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))

    else:
        trained_metapath2vec_clf, trained_metapath2vec_op, trained_metapath2vec_embedding = stt_Metapath2Vec(stellar_graph)

    # Load or generate HinSAGE Model
    # This model is good at predicting the ratings of hypothetical edges
    # BUT it does not predict the likelihood of the existence of the edge
    if os.path.exists(GRAPH_HINSAGE_MODEL) and os.path.exists(GRAPH_HINSAGE_GENERATOR):
        print("\033[91m{}\033[00m".format(f"Loading existing link prediction model (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        trained_hinsage_model = load_model(GRAPH_HINSAGE_MODEL)
        with open(GRAPH_HINSAGE_GENERATOR, 'rb') as file:
            trained_hinsage_generator = dill.load(file)
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))

    else:
        trained_hinsage_model, trained_hinsage_generator = stt_HinSAGE(stellar_graph, reviews)
    
    # Select a random user
    random_user=None
    while random_user is None:
        random_node = sample(nx_graph.nodes, 1)[0]
        if nx_graph.nodes[random_node]['type']=='user':
            random_user=random_node

    # Init for Search
    best_prediction={'probability':0,'rating':0,'product':""}

    
    print("\033[91m{}\033[00m".format(f"Searching for a good recommendation (est. ~{120} seconds)"))
    start = time.time() 

    # Starting from user's existing reviews
    users_reviews=dict(nx_graph[random_user])

    # BFS (out to depth_limit) nearby products that haven't been reviewed by the user yet
    for product in users_reviews:
        bfs = nx.bfs_tree(nx_graph,product, depth_limit=5)
        for x in bfs.nodes:     
            if nx_graph.nodes[x]['type'] == 'product' and x not in users_reviews:

                # How likely is edge to exist?
                metapath2vec_predict_features=trained_metapath2vec_op(trained_metapath2vec_embedding(random_user),trained_metapath2vec_embedding(x))
                metapath2vec_prediction=trained_metapath2vec_clf.predict_proba([metapath2vec_predict_features])
                metapath2vec_prediction = metapath2vec_prediction[0][1]
                
                # What is the likely rating of this edge?
                hin2sage_prediction_generator = trained_hinsage_generator.flow([(random_user,x)],targets=[[0]])
                hinsage_prediction = trained_hinsage_model.predict(hin2sage_prediction_generator,verbose=0)[0][0]
                
                # A 99% likely 4 is better than a 50% likely 5
                if metapath2vec_prediction*hinsage_prediction > best_prediction['probability']*best_prediction['rating']:
                    best_prediction['probability']=metapath2vec_prediction
                    best_prediction['rating']=hinsage_prediction
                    best_prediction['product']=x
                    print(f"New Best Found: {best_prediction['product']} ({best_prediction['probability']:.1%}@{best_prediction['rating']:.2f})")
                    
    print(f"\nRecommendation for {random_user}:")
    print("=User's Tastes=")
    print("\n","\n ".join(get_reviews(random_user,nx_graph,reviews)))
    
    print("=Recommended Product Info=") 
    print(f"Title: {get_title(best_prediction['product'],GRAPH_META)}")
    print("\n\t","\n\t ".join(get_categories(best_prediction['product'],GRAPH_META)))
    print(f"Likelihood of Purchase: {best_prediction['probability']}")
    print(f"Anticipated Rating: {best_prediction['rating']}")

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tSearch time: {int(end-start)}s"))
    

    


            
