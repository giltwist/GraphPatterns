import os
import simplejson
from .graph_pattern_common import parse_metadata, load_graph

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
            for i, e in enumerate(parse_metadata(GRAPH_META)):
                if i < 10:
                    print(str(i) + "|" + simplejson.dumps(e, indent=4) + "\n\n")
                else:
                    break
        else:
            print("ERROR: Graph metadata not found")
