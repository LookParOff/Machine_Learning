import numpy as np
from Algorithms.Metrics import accuracyOfBinaryClassification, getConfusionMatrix


def logisticRegression(trainData, trainDataResult, batch=32, learningRate=0.5):
    # maybe should normalize data?
    N = len(trainData)
    countOfClass = len(trainDataResult[0])
    countOfCharacteristic = len(trainData[0])
    train = trainData[:int(N * 0.85)]
    trainResult = trainDataResult[:int(N * 0.85)]
    validation = trainData[int(N * 0.85):]
    validationResult = trainDataResult[int(N * 0.85):]
    weights = np.random.random((countOfClass, countOfCharacteristic))
    biases = np.random.random((countOfClass, 1))
    iteration = 0
    model = lambda x: np.dot(weights, x) + biases
    accuracy = -1
    # while getAccuracyOfModel(model, validation, validationResult) - accuracy > 0.0001 or accuracy < 0.1:
    while iteration < 100:
        accuracy = getAccuracyOfModel(model, validation, validationResult)
        print(accuracy)
        iteration += 1
        nablaWeights = np.zeros(weights.shape)
        nablaBiases = np.zeros(biases.shape)
        for i in range(len(train)):
            x = np.reshape(train[i], (countOfCharacteristic, 1))
            correctResult = np.reshape(trainResult[i], (countOfClass, 1))
            y = np.dot(weights, x) + biases
            nablaWeights += np.dot((y - correctResult), x.transpose())
            nablaBiases += y - correctResult
        nablaWeights /= len(train)
        nablaBiases /= len(train)
        weights -= nablaWeights * learningRate
        biases -= nablaBiases * learningRate
        model = lambda x: np.dot(weights, x) + biases
    print(getAccuracyOfModel(model, validation, validationResult))
    return model


def getAccuracyOfModel(model, testInput, testOutput):
    resOfAlg = []
    correctRes = []
    for x, t in zip(testInput, testOutput):
        res = model(x)
        res = np.argmax(res)
        resOfAlg.append(res)
        correctRes.append(np.argmax(t))
    if len(testOutput[0]) == 2:
        acc = accuracyOfBinaryClassification(resOfAlg, correctRes)
        print(acc)
        return acc
    else:
        C = getConfusionMatrix(resOfAlg, correctRes)
        print(C)
        return C


N = 500
genTrainData = np.reshape([[np.random.randint(2), np.random.randint(2)] for _ in range(N)], (N, 2, 1))
genTrainRes = np.reshape([[int(x[0]==0 and x[1]==0), int(x[0]==0 and x[1]==1),
                         int(x[0]==1 and x[1]==0), int(x[0]==1 and x[1]==1)] for x in genTrainData], (N, 4, 1))
# genTrainRes = np.reshape([[int(x[0]==0), int(x[0]==1)] for x in genTrainData], (N, 2, 1))
model = logisticRegression(genTrainData, genTrainRes)
getAccuracyOfModel(model, genTrainData, genTrainRes)
print("2nd")
getAccuracyOfModel(model, genTrainData, genTrainRes)
