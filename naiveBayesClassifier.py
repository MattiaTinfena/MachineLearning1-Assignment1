from computePrior import computePrior
from computeLikelihood import computeLikelihood
from computePosterior import computePosterior


def naiveBayesClassifier(trainingSet, testSet):  

    #Compute a priori probability

    priorTraining = computePrior(trainingSet['Play'])

    #Print a priori probability

    # print("Training Set a priori probability:")
    # print(priorTraining)
    # print("\nTest Set a priori probability:")
    # print(priorTest)
    # print("\n")

    #Compute likelihood 

    likelihood = computeLikelihood(trainingSet)

    #Print likelihood

    # print("Likelihood:\n")

    # for variable, values in likelihood.items():
    #     print(f"{variable}:")
    #     for value, class_counts in values.items():
    #         rounded_class_counts = {k: round(v, 3) for k, v in class_counts.items()}
    #         print(f"  {value}: {rounded_class_counts}")


    # compute a posteriori probability

    posterioriTraining = computePosterior(trainingSet, likelihood, priorTraining)

    # print("\nA posteriori probability of the trainingSet:\n")

    # print(posterioriTraining)

    posterioriTest, prediction = computePosterior(testSet, likelihood, priorTraining)

    return posterioriTest, prediction  

