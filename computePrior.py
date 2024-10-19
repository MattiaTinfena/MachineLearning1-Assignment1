def computePrior(dataset, key):
    classes = set(dataset[key])

    n = {}

    for c in classes:
        n[c] = 0
        for v in dataset[key]:
            if v == c:
                n[c] +=1
        n[c] /= len(dataset[key])
    return n 
