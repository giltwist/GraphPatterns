import snap
import os
import re

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

    #End of dict entry found
    if colonPos == -1:
      
      #Special Cases
      if l=="discontinued product":
       entry['discontinued'] = True

      if len(entry)>0:
        if 'Id' in entry:
          yield entry
        entry = {}
      continue   

    # remove extra spaces
    l = re.sub(' +', ' ',l)

    eName = l[:colonPos]
    rest = l[colonPos+2:]
    

    # Would use switch but Python 3.9 doesn't have
    if eName == "similar":
      temp = rest.split(" ")
      if int(temp[0])>0:
        entry[eName] = temp[1:]
    elif eName == "categories":
        rows = int(rest)
        categories = []
        for i in range(rows):
          # TODO: Consider parsing categories into a tree
          l = next(f).strip()
          categories.append(l)
        entry[eName] = categories
    elif eName == "reviews":
      rows = int(rest.split(" ")[1])
      reviews = []
      for i in range(rows):
        review = {}
        l = next(f).strip()
        l = re.sub(' +', ' ',l)
        # fix persistent typo in data
        l = re.sub('cutomer', 'customer',l)
        temp = l.split(" ")
         
        review['date']=temp[0]
        for j in range(4):
          review[temp[2*j+1]]=temp[2*(j+1)] if j==0 else int(temp[2*(j+1)])
        reviews.append(review)
      entry[eName] = reviews
    
    else:
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
                print(str(i)+"|"+simplejson.dumps(e,indent=4)+"\n\n")
            else:
               break
    else:
        print("Graph metaFile not found")

