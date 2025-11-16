import os
import time
from graph_pattern_common import parse_metadata

import networkx as nx
import matplotlib.pyplot as plt


# From https://snap.stanford.edu/data/amazon0601.html
GRAPH_TEXT = "./data/amazon0601.txt"

# From https://snap.stanford.edu/data/amazon-meta.html
GRAPH_META = "./data/amazon-meta.txt"

# Graphs we are building
GRAPH_CATEGORIES = "./data/amazon-categories.txt"


if __name__ == "__main__":
    if os.path.exists(GRAPH_CATEGORIES):
        pass
    else:
        print("Category data not found...generating")

        if os.path.exists(GRAPH_META):

            graph = nx.Graph()

            all_users=[]

            start = time.time()

            for i, e in enumerate(parse_metadata(GRAPH_META)):
                #print(str(i) + "|" + simplejson.dumps(e, indent=4) + "\n\n")
                
                
                asin = e['ASIN']
                #print(i)
                if 'reviews' in e:
                    for r in e['reviews']:
                        graph.add_edge(r['customer'],asin,weight = r['rating'])
                        all_users.append(r['customer'])
                if 'categories' in e:
                    for c in e['categories']:
                        #print(c)
                        info = c.split("|")[1:]
                        for i in range(len(info)-1):
                            graph.add_edge(info[i],info[i+1])
                        graph.add_edge(info[len(info)-1],asin)
            
            #deduplicate users
            all_users=set(all_users)

            #Remove any leaf nodes
            leaves = [node for node,degree in dict(graph.degree()).items() if degree == 1]
            multi_users=all_users-set(leaves)
            print(len(multi_users))
            graph.remove_nodes_from(leaves)

            # Only for small numbers of nodes
            #plt.figure(figsize=(60, 60)) 
            #nx.draw(graph, with_labels=True, node_size=50)
            #plt.savefig("test.png")

            end = time.time()

            print("\tElapsed time: ", end-start)

            
