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


def accuracyOfBinaryClassification(resultOfAlgorithm, answer):
    TP, TN, FP, FN = getMetrics(resultOfAlgorithm, answer)
    return (TP + TN) / len(resultOfAlgorithm)


def alfaBettaError(resultOfAlgorithm, answer):
    TP, TN, FP, FN = getMetrics(resultOfAlgorithm, answer)
    alfa = FP / (FP + TN)  # probability, that element of 0 class, algorithm recognize as 1 class
    betta = FN / (FN + TP)  # probability, that element of 1 class, algorithm recognize as 0 class
    return alfa, betta


def getPrecisionAndRecall(resultOfAlgorithm, answer):
    TP, TN, FP, FN = getMetrics(resultOfAlgorithm, answer)
    precision = TP / (TP + FP)  # which part algorithm recognize as 1 class and it's correct
    recall = TP / (TP + FN)  # which part algorithm recognize as 1 class of all elements of 1 class
    return precision, recall


def getFScore(resultOfAlgorithm, answer):
    precision, recall = getPrecisionAndRecall(resultOfAlgorithm, answer)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1

