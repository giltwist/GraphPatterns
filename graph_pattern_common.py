import re
from math import ceil, log2

import networkx as nx

# == GLOBAL CONFIG ==

# NOTE: Accepts one in GRAPH_REDUCTION_FACTOR products from the original dataset to accommodate system memory
# Uncomment only one line
# For rapid iterating on code, a factor of 100 is recommended
#GRAPH_REDUCTION_FACTOR = 100
# For a quick run, a factor of 10 is recommended
#GRAPH_REDUCTION_FACTOR = 10
# With 16 GB of memory, factor of 4 is recommended
GRAPH_REDUCTION_FACTOR = 4
# If you have 32 or 64GB of memory, a factor of 1 is potentially achievable
#GRAPH_REDUCTION_FACTOR = 1


# Original Datasource
# From https://snap.stanford.edu/data/amazon-meta.html
GRAPH_META = "./data/amazon-meta.txt"

# General Graph File Locations
GRAPH_CATEGORIES = f"./data/amazon-categories_GRF{GRAPH_REDUCTION_FACTOR}.pkl"
GRAPH_VECTORIZER = f'./data/vectorizer-GRF{GRAPH_REDUCTION_FACTOR}.pkl'
GRAPH_REVIEWSDF = f"./data/amazon-reviewsdf_GRF{GRAPH_REDUCTION_FACTOR}.pkl"

# HinSAGE File Locations
GRAPH_HINSAGE_MODEL = f"./data/hinsage-model_GRF{GRAPH_REDUCTION_FACTOR}"
GRAPH_HINSAGE_GENERATOR = f"./data/hinsage-generator_GRF{GRAPH_REDUCTION_FACTOR}.pkl"

# Metapath2Vec File Locations
GRAPH_METAPATH2VEC_MODEL = f"./data/metapath2vec-model_GRF{GRAPH_REDUCTION_FACTOR}"


# == END CONFIG == 

def get_title(asin, filepath):
    try:
        with open(filepath, "r") as f:
            entry = {}
            for l in f:
                l = l.strip()
                if l == f"ASIN: {asin}":
                    title = next(f).strip()[7:]
                    return title
            return "Not Found"
                            
    except Exception as e:
        print(e)

def get_categories(asin, filepath):
    try:
        with open(filepath, "r") as f:
            entry = {}
            for l in f:
                l = l.strip()
                if l == f"ASIN: {asin}":
                    l=next(f).strip()
                    while not l.startswith("categories") and len(l)>0:
                        l =next(f).strip()
                    if len(l)==0:
                        return []
                    else:
                        colonPos = l.find(":")
                        rows = int(l[colonPos + 2 :])
                        if rows>0:
                            categories = []
                            for i in range(rows):
                                l = next(f).strip()
                                categories.append(l)
                            return categories
            return []
                            
    except Exception as e:
        print(e)


# adapted from https://snap.stanford.edu/data/web-Amazon.html
def parse_metadata(filepath):
    try:
        with open(filepath, "r") as f:
            entry = {}
            for l in f:
                l = l.strip()

                colonPos = l.find(":")

                # End of dict entry found
                if colonPos == -1:

                    # Special Cases
                    if l == "discontinued product":
                        entry["discontinued"] = True

                    if len(entry) > 0:
                        if "Id" in entry and "ASIN" in entry:
                            yield entry
                        entry = {}
                    continue

                # remove extra spaces
                l = re.sub(" +", " ", l)

                eName = l[:colonPos]
                rest = l[colonPos + 2 :]

                if eName == "similar":
                    temp = rest.split(" ")
                    if int(temp[0]) > 0:
                        entry[eName] = temp[1:]
                elif eName == "categories":
                    rows = int(rest)
                    categories = []
                    if rows>0:
                        for i in range(rows):
                            l = next(f).strip()
                            categories.append(l)
                        entry[eName] = categories
                elif eName == "reviews":
                    rows = int(rest.split(" ")[3])
                    reviews = []
                    #print(rows)
                    if rows>0:
                        for i in range(rows):
                            review = {}
                            l = next(f).strip()
                            l = re.sub(" +", " ", l)
                            # fix persistent typo in data
                            l = re.sub("cutomer", "customer", l)
                            temp = l.split(" ")

                            review["date"] = temp[0]
                            for j in range(4):
                                review[temp[2 * j + 1][:-1]] = (
                                    temp[2 * (j + 1)] if j == 0 else int(temp[2 * (j + 1)])
                                )
                            reviews.append(review)
                        
                        entry[eName] = reviews
                else:
                    entry[eName] = rest
            if "Id" in entry and "ASIN" in entry:
                yield entry
    except Exception as e:
        print(e)

#Must be NetworkX Graph
# Most products have 1-10 reviews, but some have thousands, this can skew NN training
# This function reduces that skew while keeping some proportionality to degree of products
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
