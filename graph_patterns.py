from graph_pattern_common import parse_metadata, get_title
from graph_pattern_common import GRAPH_REDUCTION_FACTOR, GRAPH_META, GRAPH_MODEL, GRAPH_CATEGORIES
from graph_NN import split_train_test


import os
import time
import networkx as nx
import numpy as np

from math import ceil, log2

import matplotlib.pyplot as plt

from stellargraph import StellarGraph

import dill

from random import sample

# NetworkX has nicer building and storing functions for graphs than StellarGraph
def generate_graph():

    if os.path.exists(GRAPH_META):

        graph = nx.Graph()

        #all_users=[]


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
                        #Only append positive ratings as this model doesn't handle negative ratings
                        if r['rating']>=3:
                            graph.add_node(r['customer'],type='user')
                            graph.add_edge(r['customer'],asin,weight = r['rating'], type='review')
                            #all_users.append(r['customer'])
                if 'categories' in e:
                    for c in e['categories']:
                        #print(c)
                        info = c.split("|")[1:]
                        for i in range(len(info)-1):
                            graph.add_node(info[i],type='category')
                            graph.add_node(info[i+1],type='category')
                            graph.add_edge(info[i],info[i+1],weight = i+1, type='category_hierarchy')
                        graph.add_edge(info[len(info)-1],asin,weight = len(info), type='product_type')
                #TODO: Add product-to-product edges with the similar-to information.
        
        #deduplicate users
        #all_users=set(all_users)
        #print(f"Total users: {len(all_users)}")

        #Remove any leaf nodes
        #print("Pruning singleton reviewers.")
        #leaves = [node for node,degree in dict(graph.degree()).items() if degree == 1]
        #multi_users=all_users-set(leaves)
        #print(f"Reviewers with multiple reviews: {len(multi_users)}.")
        #graph.remove_nodes_from(leaves)


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

def estimate_progress(task,duration):
    pass

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
        
#Must be NetworkX Graph
# Most products have 1-10 reviews, but some have thousands, this skews NN training
# This function reduces that skew while keeping some weighting
def unskew_graph(graph):
    print(f"Before unskewing: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    for node in graph.nodes:
        if graph.nodes[node]['type']=='product':
            reviewers = []
            neighbors=dict(graph[node])
            for neighbor in neighbors:
                if graph.nodes[neighbor]['type']=='user':
                   reviewers.append((neighbor,graph.degree[neighbor]))
            # All reviewers in ascending order of degree
            reviewers.sort(key=lambda x: x[1])
            # 1000 reviews becomes 10, 10 reviews becomes 4
            if len(reviewers)>1:
                to_keep = int(ceil(log2(len(reviewers))))
                # remove edges from all but those of the highest degree
                for r in reviewers[:-1*to_keep]:
                    graph.remove_edge(node,r[0])
    # Remove any nodes that became isolated as a result of this process
    graph.remove_nodes_from(list(nx.isolates(graph)))
    print(f"After unskewing: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph
              
# NOTE: Time estimates were derived on an AMD 3770X CPU with 16GB memory
if __name__ == "__main__":

    

    # Loading is much faster than generating, particular with low GRAPH_REDUCTION_FACTOR


    nx_graph = nx.Graph()
    unskewed_graph = nx.Graph()
    trained=None

    # Try to load at least the categories data
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
        # Generate everything
        print("\033[91m{}\033[00m".format(f"Generating new category graph (est. ~90 seconds)"))
        start = time.time()
        nx_graph = generate_graph()
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tGeneration time: {int(end-start)}s"))

    unskewed_graph = unskew_graph(nx_graph)
    # Try to load complete model
    if os.path.exists(GRAPH_MODEL):
        print("\033[91m{}\033[00m".format(f"Loading existing link prediction model (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        with open(GRAPH_MODEL, 'rb') as file:
            trained = dill.load(file)
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tLoading time: {int(end-start)}s"))

    else:
        # Send NetworkX graph to StellarGraph format


        print("\033[91m{}\033[00m".format(f"Activating StellarGraph Library (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
        start = time.time()
        stellar_graph = StellarGraph.from_networkx(unskewed_graph,node_type_attr='type',edge_type_attr='type', edge_weight_attr='weight')
        print(stellar_graph.info())
        end = time.time()
        print("\033[93m{}\033[00m".format(f"\tActivation time: {int(end-start)}s"))

        # Do actual training of GNN using Metapath2Vec
        trained = split_train_test(stellar_graph)
    
    
    trained_clf = trained['classifier']
    trained_op = trained['binary_operator']
    trained_embedding = trained['embedding']

    # Predict if a random user will like a random product
    random_user=None

    while random_user is None:
        random_node = sample(unskewed_graph.nodes, 1)[0]
        if unskewed_graph.nodes[random_node]['type']=='user':
            random_user=random_node

    best_prediction=0
    best_product=None

    users_reviews=dict(unskewed_graph[random_user])

    print("\033[91m{}\033[00m".format(f"Searching for a good recommendation (est. ~{ceil(600/GRAPH_REDUCTION_FACTOR)} seconds)"))
    start = time.time() 
    for product in users_reviews:
        bfs = nx.bfs_tree(unskewed_graph,product, depth_limit=10)
        for x in bfs.nodes:     
            if unskewed_graph.nodes[x]['type'] == 'product' and x not in users_reviews:
                predict_features=trained_op(trained_embedding(random_user),trained_embedding(x))
                prediction=trained_clf.predict_proba([predict_features])
                #print(prediction)
                if prediction[0][1] > best_prediction:
                    best_prediction=prediction[0][1]
                    best_product=x
                    print(f"{best_product} > {prediction}")
    
    print(f"\nRecommendation for {random_user}:")
    print("=User's Tastes=")
    print("\n","\n ".join(get_reviews(random_user,nx_graph)))
    print("=Recommended Product Info=")
    print(f"Title: {get_title(best_product,GRAPH_META)}")
    print("\n\t","\n\t ".join(get_types(best_product,nx_graph)))
    print(f"Confidence in prediction: {best_prediction:.1%}")

    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tSearch time: {int(end-start)}s"))
    


        
        

    #visualize_graph(graph)

            
