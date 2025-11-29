from graph_pattern_common import parse_metadata, get_title
from graph_pattern_common import GRAPH_REDUCTION_FACTOR, GRAPH_META, GRAPH_MODEL, GRAPH_CATEGORIES
from graph_NN import split_train_test


import os
import time
import networkx as nx
import numpy as np

from math import ceil

import matplotlib.pyplot as plt

from stellargraph import StellarGraph



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
                            graph.add_edge(info[i],info[i+1],weight = i+1, type='category_hierarchy')
                        graph.add_edge(info[len(info)-1],asin,weight = len(info), type='product_type')
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



def estimate_progress(task,duration):
    pass

def get_reviews(user, graph):
    neighbors=dict(graph[user])
    for neighbor in neighbors:
        print(neighbor,neighbors[neighbor]['weight'], get_categories(neighbor, graph))

def get_categories(node, graph):

    neighbors = dict(graph[node])
    for neighbor in neighbors:
        if neighbors[neighbor]['type']=="product_type":
            return get_categories(neighbor,graph)
        
    



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

    # TESTING AREA

    done= False

    while not done:
        random_node = sample(nx_graph.nodes, 1)[0]      
        if nx_graph.nodes[random_node]['type']=='user':
            get_reviews(random_node,nx_graph)
            done=True
            
    # Send NetworkX graph to StellarGraph format

    print("\033[91m{}\033[00m".format(f"Activating StellarGraph Library (est. ~{ceil(60/GRAPH_REDUCTION_FACTOR)} seconds)"))
    start = time.time()
    stellar_graph = StellarGraph.from_networkx(nx_graph,node_type_attr='type',edge_type_attr='type', edge_weight_attr='weight')
    print(stellar_graph.info())
    end = time.time()
    print("\033[93m{}\033[00m".format(f"\tActivation time: {int(end-start)}s"))


    # Do actual training of GNN using Metapath2Vec
    trained = split_train_test(stellar_graph)
    trained_clf = trained['classifier']
    trained_op = trained['binary_operator']
    trained_embedding = trained['embedding']

    # Predict if a random user will like a random product
    for i in range(10):
        random_user=None
        random_product=None

        while (random_user is None or random_product is None):

            random_node = sample(nx_graph.nodes, 1)[0]
            
            if random_user is None and nx_graph.nodes[random_node]['type']=='user':
                random_user=random_node
            elif random_product is None and nx_graph.nodes[random_node]['type'] == 'product':
                random_product=random_node

        
        print(f"Predicting on {random_user} -> {random_product}")
        predict_link=np.array([trained_embedding(random_user),trained_embedding(random_product)])
        print(trained_clf.predict_proba(predict_link))
        


        
        
        

    #visualize_graph(graph)

            
