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

def generate_graph():

    if os.path.exists(GRAPH_META):

        graph = nx.Graph()

        all_users=[]

        start = time.time()

        for i, e in enumerate(parse_metadata(GRAPH_META)):
            #print(str(i) + "|" + simplejson.dumps(e, indent=4) + "\n\n")
            
            
            asin = e['ASIN']
            #print(i)
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
        
        #deduplicate users
        all_users=set(all_users)
        print(f"Total users: {len(all_users)}")

        #Remove any leaf nodes
        print("Pruning singleton reviewers.")
        leaves = [node for node,degree in dict(graph.degree()).items() if degree == 1]
        multi_users=all_users-set(leaves)
        print(f"Reviewers with multiple reviews: {len(multi_users)}.")
        graph.remove_nodes_from(leaves)

        end = time.time()

        nx.write_graphml(graph,GRAPH_CATEGORIES)
        print("\tElapsed time: ", end-start)
        return graph
    else:
        print("Amazon metadata not found.  Download it from  https://snap.stanford.edu/data/amazon-meta.html")


def visualize_graph(graph):
    # Only for small numbers of nodes
    plt.figure(figsize=(60, 60)) 
    if (graph.number_of_nodes()<1000 and graph.number_of_edges()>1000):
        
        nx.draw(graph, with_labels=True, node_size=50)
    else:
        print("Graph too large to visualize, summarizing")
        summary_graph=nx.snap_aggregation(graph, node_attributes=("type",))
        nx.draw(summary_graph, with_labels=True, node_size=50)
    plt.savefig("test.png")



if __name__ == "__main__":
    if os.path.exists(GRAPH_CATEGORIES):
        print("Category graph found")
        graph = nx.read_edgelist(GRAPH_CATEGORIES,nodetype=str, data=(('weight',int),('color',str)))
    else:
        print("Category graph not found...generating")
        graph = generate_graph()
    
    visualize_graph(graph)

            
