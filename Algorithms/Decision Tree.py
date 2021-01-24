import numpy as np
import pandas as pd
from Algorithms.Metrics import getPrecisionAndRecall, getFScore


class Tree:
    def __init__(self):
        self.tau = 0
        self.indexOfSearchFeature = 0
        self.depth = 0
        self.leaf = False
        self.leafLabel = None
        self.parent = None
        self.leftSon = None
        self.rightSon = None

    def evaluate(self, x):
        if not self.leaf:
            if x[self.indexOfSearchFeature] < self.tau:
                return self.leftSon.evaluate(x)
            else:
                return self.rightSon.evaluate(x)
        else:
            return self.leafLabel

    def createLeftSon(self):
        self.leftSon = Tree()
        self.leftSon.parent = self
        # self.leftSon.tau = tau
        self.leftSon.depth = self.depth + 1

    def createRightSon(self):
        self.rightSon = Tree()
        self.rightSon.parent = self
        # self.rightSon.tau = tau
        self.rightSon.depth = self.depth + 1

    def getLeftSon(self):
        return self.leftSon

    def getRightSon(self):
        return self.rightSon

    def __str__(self):
        printArray[self.depth].append((self.tau, self.indexOfSearchFeature))
        self.leftSon.__str__()
        self.rightSon.__str__()


printArray = [[] for _ in range(100)]


def getEntropy(Ni, classes):
    # Ni- len of selection
    # countsOfEachClass+1 is just little trick to escape log2(0) = -inf. Value of Entropy is still ~correct
    countsOfEachClass = np.bincount(classes)
    entropy = -np.sum((countsOfEachClass / Ni) * np.log2((countsOfEachClass + 0.000001) / Ni))
    return entropy


def lossFunction(arrOfDirForEachX, classesForEachX):
    # arrOfDir- True or False for each element of selection, True or False means we go to the left or right son
    # Ni- len of current selection, which went to this edge
    Ni = len(arrOfDirForEachX)
    Ni1 = np.count_nonzero(arrOfDirForEachX)
    Ni0 = len(arrOfDirForEachX) - Ni1

    # entropyOfSi0 = Ni0 / Ni * np.log2(Ni0 / Ni+0.000001)
    # entropyOfSi1 = Ni1 / Ni * np.log2(Ni1 / Ni+0.000001)
    entropyOfSi0 = getEntropy(Ni, classesForEachX[arrOfDirForEachX == False])
    entropyOfSi1 = getEntropy(Ni, classesForEachX[arrOfDirForEachX])
    result = Ni0 / Ni * entropyOfSi0 + Ni1 / Ni * entropyOfSi1
    return result


def decisionTree(trainData, trainDataResult, edgeOfTree, countOfClasses,
                 numberOfFeature=1, maximumDepth=25, minPowerOfSelection=1, minEntropy=0.05):
    # depth- how deep tree can be
    # minCountOfSelection- if selection lower than this value- we make leaf
    # minEntropy- if Entropy we gain after split the selection lower than this value- we make leaf
    # numberOfFeature- how much Feature selection function will return to us parameters
    entropy = getEntropy(len(trainData), trainDataResult)
    print(entropy)
    if edgeOfTree.depth > maximumDepth or len(trainData) <= minPowerOfSelection or entropy < minEntropy:
        edgeOfTree.leaf = True
        t = np.bincount(trainDataResult, minlength=countOfClasses)
        edgeOfTree.leafLabel = t / len(trainDataResult)
        return
    countOfCharacteristic = len(trainData[0])
    lossForEachFeature = []
    tauForCharacteristic = []
    for indexOfCharacteristic in range(countOfCharacteristic):
        xs = trainData[:, indexOfCharacteristic]
        allTau = np.arange(min(xs), max(xs), 0.1)  # all possible threshold values
        if len(allTau) == 0:
            # xs- array with just one characteristic
            lossForEachFeature.append(np.nan)
            tauForCharacteristic.append(np.nan)
            continue
        w, v = np.meshgrid(xs, allTau)
        arrayOfDirections = w >= v  # direction left or right son
        # i element is how much of selection goes to the right son with allTau[i]
        # countOfRightElements = np.count_nonzero(arrayOfDirections, axis=1)
        allPossibleValuesLossFunctions = np.apply_along_axis(lossFunction, 1, arrayOfDirections, trainDataResult) # apply to each cols not to each element
        indexOfMinLoss = np.argmin(allPossibleValuesLossFunctions)  # index of tau with minimum loss
        lossForEachFeature.append(np.min(allPossibleValuesLossFunctions))
        tauForCharacteristic.append(allTau[indexOfMinLoss])
    if len(np.argwhere(np.isnan(lossForEachFeature))) == len(lossForEachFeature):
        # every feature vector consist of just one value
        edgeOfTree.leaf = True
        t = np.bincount(trainDataResult, minlength=countOfClasses)
        edgeOfTree.leafLabel = t / len(trainDataResult)
        return
    indexOfFeature = np.nanargmin(lossForEachFeature)
    resultTau = tauForCharacteristic[indexOfFeature]
    if np.all(trainData[:, indexOfFeature] < resultTau) or np.all(trainData[:, indexOfFeature] >= resultTau):
        # our tau does not change anything, all selection goes to the left or right. It's useless
        edgeOfTree.leaf = True
        t = np.bincount(trainDataResult, minlength=countOfClasses)
        edgeOfTree.leafLabel = t / len(trainDataResult)
        return
    edgeOfTree.tau = resultTau
    edgeOfTree.indexOfSearchFeature = indexOfFeature

    edgeOfTree.createLeftSon()
    # print(trainDataResult[np.nonzero(trainData[:, indexOfCharacteristic] > resultTau)[0]])
    decisionTree(trainData[trainData[:, indexOfFeature] < resultTau], trainDataResult[np.nonzero(trainData[:, indexOfFeature] < resultTau)[0]],
                 edgeOfTree.getLeftSon(), countOfClasses, numberOfFeature, maximumDepth, minPowerOfSelection, minEntropy)

    edgeOfTree.createRightSon()
    decisionTree(trainData[trainData[:, indexOfFeature] >= resultTau], trainDataResult[np.nonzero(trainData[:, indexOfFeature] >= resultTau)[0]],
                 edgeOfTree.getRightSon(), countOfClasses, numberOfFeature, maximumDepth, minPowerOfSelection, minEntropy)
    return 0


def trainDecisionTreeClassification(trainData, trainDataResult,
                                    numberOfFeature=1, maximumDepth=10, minPowerOfSelection=1, minEntropy=0.05):
    root = Tree()
    countOfClasses = len(set(trainDataResult))
    decisionTree(trainData, trainDataResult, root, countOfClasses, numberOfFeature, maximumDepth, minPowerOfSelection, minEntropy)
    return root


# read CSV
df = pd.read_csv("../Datasets/Titanic/train.csv")
df["Sex"].replace("female", 0, inplace=True)
df["Sex"].replace("male", 1, inplace=True)
sex = np.reshape(list(df["Sex"]), (len(df["Sex"]), 1))
Pclass = np.reshape(list(df["Pclass"]), (len(df["Pclass"]), 1))
InputVectorsOfPeople = np.concatenate((sex, Pclass), axis=1)
OutputClasses = np.array(df["Survived"], dtype=np.int32)
model = trainDecisionTreeClassification(InputVectorsOfPeople, OutputClasses)

resOfAlg = [np.argmax(model.evaluate(x)) for x in InputVectorsOfPeople]
print("PRECISION and RECALL", getPrecisionAndRecall(resOfAlg, OutputClasses))
print("F-SCORE", getFScore(resOfAlg, OutputClasses))

# make result CSV
df = pd.read_csv("../Datasets/titanic/test.csv")
df["Sex"].replace("female", 0, inplace=True)
df["Sex"].replace("male", 1, inplace=True)
sex = np.reshape(list(df["Sex"]), (len(df["Sex"]), 1))
Pclass = np.reshape(list(df["Pclass"]), (len(df["Pclass"]), 1))
InputVectorsOfPeople = np.concatenate((sex, Pclass), axis=1)

outputlist = []
for passId, man in zip(df["PassengerId"], InputVectorsOfPeople):
    outputlist.append([passId, np.argmax(model.evaluate(man))])
dfTest = pd.DataFrame(outputlist, columns=['PassengerId', 'Survived'])
# dfTest.to_csv('submission.csv', index=False)
print(dfTest)

