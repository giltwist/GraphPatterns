import re


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
                        if "Id" in entry:
                            yield entry
                        entry = {}
                    continue

                # remove extra spaces
                l = re.sub(" +", " ", l)

                eName = l[:colonPos]
                rest = l[colonPos + 2 :]

                # Would use switch but Python 3.9 doesn't have
                if eName == "similar":
                    temp = rest.split(" ")
                    if int(temp[0]) > 0:
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
                        l = re.sub(" +", " ", l)
                        # fix persistent typo in data
                        l = re.sub("cutomer", "customer", l)
                        temp = l.split(" ")

                        review["date"] = temp[0]
                        for j in range(4):
                            review[temp[2 * j + 1]] = (
                                temp[2 * (j + 1)] if j == 0 else int(temp[2 * (j + 1)])
                            )
                        reviews.append(review)
                    entry[eName] = reviews

                else:
                    entry[eName] = rest
            yield entry
    except:
        print("ERROR: File not found")
