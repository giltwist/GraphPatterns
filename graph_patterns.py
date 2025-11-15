import snap
import os

import simplejson



GRAPH_TEXT="./data/amazon0601.txt"
GRAPH_META="./data/amazon-meta.txt"

def make_graph():
    G = snap.LoadEdgeListStr(snap.TNGraph, GRAPH_TEXT, 0, 1)
    print("Number of Nodes: %d" % G.GetNodes())

    G = snap.LoadEdgeListStr(snap.TUNGraph, GRAPH_TEXT, 0, 1)
    print("Number of Nodes: %d" % G.GetNodes())

    G = snap.LoadEdgeListStr(snap.TNEANet, GRAPH_TEXT, 0, 1)
    print("Number of Nodes: %d" % G.GetNodes())

    (G, Map) = snap.LoadEdgeListStr(snap.TNGraph, GRAPH_TEXT, 0, 1, True)
    print("Number of Nodes: %d" % G.GetNodes())
    print("Number of Nodes: %d" % Map.Len())

    # convert input string to node id
    NodeId = Map.GetKeyId("1065")
    # convert node id to input string
    NodeName = Map.GetKey(NodeId)
    print("name", NodeName)
    print("id  ", NodeId)

# adapted from https://snap.stanford.edu/data/web-Amazon.html
def parse(filename):
  f = open(filename, 'r')
  entry = {}
  for l in f:
    l = l.strip()
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry



if __name__ == "__main__":
    if os.path.exists(GRAPH_TEXT):
        #make_graph()
        pass
    else:
        print("Graph textFile not found")

    if os.path.exists(GRAPH_META):
        for i,e in enumerate(parse(GRAPH_META)):
            if i<10:
                print(simplejson.dumps(e))
            else:
               break
    else:
        print("Graph metaFile not found")

