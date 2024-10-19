from computePrior import computePrior
from computeLikelihoodLaplace import computeLikelihoodLaplace
from computePosterior import computePosterior


def naiveBayesClassifierLaplace(trainingSet, testSet, alpha):
   
    #Compute a priori probability

    priorTraining = computePrior(trainingSet, "Play")

    #Print a priori probability

    # print("Training Set a priori probability:")
    # print(priorTraining)
    # print("\n")

    #Compute likelihood 

    likelihood = computeLikelihoodLaplace(trainingSet, "Play", alpha)

    #Print likelihood Laplace

    print("Likelihood Laplace:\n")

    for variable, values in likelihood.items():
        print(f"{variable}:")
        for value, class_counts in values.items():
            rounded_class_counts = {k: round(v, 3) for k, v in class_counts.items()}
            print(f"  {value}: {rounded_class_counts}")


    #Compute a posteriori probability and predictions

    posterioriTest, predictions = computePosterior(testSet, likelihood, priorTraining)

    #Print a posteriori probability and predictions

    print("\n")
    print(f"{'yes':<10}{'no':<10}{'predictions Laplace':<15}")
    print("-" * 35)
    
    for i in range(len(testSet[next(iter(testSet))])):
        yes_prob = posterioriTest['yes'][i] if 'yes' in posterioriTest else 0
        no_prob = posterioriTest['no'][i] if 'no' in posterioriTest else 0
        prediction = predictions[i] if i < len(predictions) else "N/A"

        print(f"{yes_prob:<10.4f}{no_prob:<10.4f}{prediction:<15}")
        print("\n")

    return posterioriTest, predictions