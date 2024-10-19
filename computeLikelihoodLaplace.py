def computeLikelihoodLaplace(dataset, key, alpha):
    valueCount = {}
    classes = set(dataset[key])

    v = {variable: len(set(dataset[variable])) for variable in dataset if variable != key}

    for variable in dataset:
        if variable == key:
            continue  
        valueCount[variable] = {}
        for i, value in enumerate(dataset[variable]):
            class_label = dataset[key][i]
            if value not in valueCount[variable]:
                valueCount[variable][value] = {c: 0 for c in classes}
            valueCount[variable][value][class_label] += 1
    class_counts = {c: dataset[key].count(c) for c in classes}

    for variable in valueCount:
        for value in valueCount[variable]:
            for c in classes:
                count = valueCount[variable][value][c]
                if count == 0:
                    valueCount[variable][value][c] = (count + alpha) / (class_counts[c] + alpha * v[variable])
                else:
                    valueCount[variable][value][c] = count / class_counts[c]

    return valueCount
