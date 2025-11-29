import re

# == GLOBAL CONFIG ==

# NOTE: Accepts one in GRAPH_REDUCTION_FACTOR products from the original dataset to accommodate system memory
# Uncomment only one line
# For rapid iterating on variables, a factor of 100 is recommended
GRAPH_REDUCTION_FACTOR = 100
# For a quick run, a factor of 10 is recommended
#GRAPH_REDUCTION_FACTOR = 10
# With 16 GB of memory, factor of 3 is recommended
# GRAPH_REDUCTION_FACTOR = 3
# If you have 32 or 64GB of memory, a factor of 1 is potentially achievable
#GRAPH_REDUCTION_FACTOR = 1


# From https://snap.stanford.edu/data/amazon-meta.html
GRAPH_META = "./data/amazon-meta.txt"

# Graph we are building
GRAPH_CATEGORIES = f"./data/amazon-categories_GRF{GRAPH_REDUCTION_FACTOR}.txt"

# Model we are building

GRAPH_MODEL = f"./data/amazon-prediction_GRF{GRAPH_REDUCTION_FACTOR}.pkl"


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
