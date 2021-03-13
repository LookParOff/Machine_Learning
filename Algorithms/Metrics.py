import numpy as np


def getMetrics(resultOfAlgorithm, answer):
    TP = 0  # Correct! First class
    TN = 0  # Correct! Zero class
    FP = 0  # Wrong! It's- first class
    FN = 0  # Wrong! It's- zero class
    for res, answ in zip(resultOfAlgorithm, answer):
        if res == answ == 1:
            TP += 1
        if res == answ == 0:
            TN += 1
        if res == 1 and answ == 0:
            FN += 1
        if res == 0 and answ == 1:
            FP += 1
    return TP, TN, FP, FN


def getConfusionMatrix(resultOfAlgorithm, answer):
    # resultOfAlgorithm- class of i object element in range [0, countOfClass+1] on opinion of algorithm
    # answer- class of i object element in range [0, countOfClass+1] in reality
    countOfClass = max(answer) + 1
    confMat = np.zeros((countOfClass, countOfClass))
    for res, answ in zip(resultOfAlgorithm, answer):
        confMat[answ][res] += 1
    return confMat


def accuracyOfBinaryClassification(resultOfAlgorithm, answer):
    TP, TN, FP, FN = getMetrics(resultOfAlgorithm, answer)
    return (TP + TN) / len(resultOfAlgorithm)


def accuracyOfClassification(resultOfAlgorithm, answer):
    correct = 0
    for res, answ in zip(resultOfAlgorithm, answer):
        if res == answ:
            correct += 1
    return correct / len(resultOfAlgorithm)


def alfaBettaError(resultOfAlgorithm, answer):
    TP, TN, FP, FN = getMetrics(resultOfAlgorithm, answer)
    alfa = FP / (FP + TN)  # probability, that element of 0 class, algorithm recognize as 1 class
    betta = FN / (FN + TP)  # probability, that element of 1 class, algorithm recognize as 0 class
    return alfa, betta


def getPrecisionAndRecall(resultOfAlgorithm, answer):
    TP, TN, FP, FN = getMetrics(resultOfAlgorithm, answer)
    try:
        precision = TP / (TP + FP)  # which part algorithm recognize as 1 class and it's correct
        recall = TP / (TP + FN)  # which part algorithm recognize as 1 class of all elements of 1 class
    except ZeroDivisionError:
        return 0, -1
    return precision, recall


def getFScore(resultOfAlgorithm, answer):
    precision, recall = getPrecisionAndRecall(resultOfAlgorithm, answer)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1


def getMeanSquaredError(resultOfAlgorithm, answer):
    result = 0
    for alg, answ in zip(resultOfAlgorithm, answer):
        result += (answ - alg) ** 2
    return result / len(answer)

