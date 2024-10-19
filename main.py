from naiveBayesClassifier import naiveBayesClassifier
from naiveBayesClassifierLaplace import naiveBayesClassifierLaplace
from LoadData import loadData
from printResults import printResults
tsLength = 10
alpha = 1

data_dict, trainingSet, testSet= loadData('weatherData.txt',tsLength)  

#Trainingset & testset print

# df_trainingSet = pd.DataFrame(trainingSet)
# df_testSet = pd.DataFrame(testSet)
# print("\nTraining Set:\n")
# print(df_trainingSet.to_string(index=False)) 
# print("\nTest Set:\n")
# print(df_testSet.to_string(index=False))
# print("\n")

if len(testSet) == len(trainingSet):
    
    errorRate = []
    errorRateLaplace = []

    num_iterations = 1000

    for _ in range(num_iterations):

        data_dict, trainingSet, testSet= loadData('weatherData.txt',tsLength)  
        posterioriProb, prediction = naiveBayesClassifier(trainingSet, testSet)
        posterioriProbLaplace, predictionLaplace = naiveBayesClassifierLaplace(trainingSet, testSet, alpha)

        res = 0
        resLaplace = 0

        for i in range(len(prediction)):  
            if testSet['Play'][i] != prediction[i]:  
                res += 1
            if testSet['Play'][i] != predictionLaplace[i]:  
                resLaplace += 1

        res /= len(prediction)
        resLaplace /= len(predictionLaplace)

        errorRate.append(res)
        errorRateLaplace.append(resLaplace)   

    printResults(errorRate, errorRateLaplace, num_iterations)

else:
    posterioriProb, prediction = naiveBayesClassifier(trainingSet, testSet)
    posterioriProbLaplace, predictionLaplace = naiveBayesClassifierLaplace(trainingSet, testSet, alpha)

    print("print predictions of the classifier\n")
    print(prediction)
    print("\nprint predictions of the classifier with Laplace smoothing\n")
    print(predictionLaplace)


    