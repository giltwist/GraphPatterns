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
                        if "Id" in entry and "ASIN" in entry:
                            yield entry
                        entry = {}
                    continue

                # remove extra spaces
                l = re.sub(" +", " ", l)

                eName = l[:colonPos]
                rest = l[colonPos + 2 :]

                match eName:
                    case "similar":
                        temp = rest.split(" ")
                        if int(temp[0]) > 0:
                            entry[eName] = temp[1:]
                    case "categories":
                        rows = int(rest)
                        categories = []
                        if rows>0:
                            for i in range(rows):
                                l = next(f).strip()
                                categories.append(l)
                            entry[eName] = categories
                    case "reviews":
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
                    case _:
                        entry[eName] = rest
            if "Id" in entry and "ASIN" in entry:
                yield entry
    except Exception as e:
        print(e)
