import os
import time
from graph_pattern_common import parse_metadata

import networkx as nx
from stellargraph import StellarGraph
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


        # can use iterator i for limiting loop
        # otherwise use entry e for data access
        for i, e in enumerate(parse_metadata(GRAPH_META)):
            
            #Useful for debugging
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


        # NOTE: GraphML stored node types as well, but was 1.1GB instead of 300MB
        # NOTE: Switched to edge list based on research that suggested it was more memory efficient
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



if __name__ == "__main__":
    if os.path.exists(GRAPH_CATEGORIES):
        print("Category graph found")

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
        print("Loading time: ", end-start)
    else:
        print("Category graph not found...generating")
        start = time.time()
        nx_graph = generate_graph()
        end = time.time()
        print("Generation time: ", end-start)

    print('Activating StellarGraph Library')
    stellar_graph = StellarGraph.from_networkx(nx_graph,node_type_attr='type',edge_type_attr='type')
    print(stellar_graph.info())

    #visualize_graph(graph)

            
