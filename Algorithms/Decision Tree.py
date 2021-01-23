import numpy as np


class Tree:
    def __init__(self):
        self.tau = 0
        self.indexOfSearchCharacteristic = 0
        self.depth = 0
        self.leaf = False
        self.leafLabel = None
        self.parent = None
        self.leftSon = None
        self.rightSon = None

    def evaluate(self, x):
        if not self.leaf:
            if x[self.indexOfSearchCharacteristic] > self.tau:
                self.rightSon.evaluate(x)
            else:
                self.leftSon.evaluate(x)
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
        return self.leftSon

    # def __str__(self):
    #     return self.leftSon.__str__() + "\n" + str(self.selection) + str(self.classes) + "\n" + self.rightSon.__str__()


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
    entropyOfSi0 = getEntropy(Ni0, classesForEachX)
    entropyOfSi1 = getEntropy(Ni1, classesForEachX)
    result = Ni0 / Ni * entropyOfSi0 + Ni1 / Ni * entropyOfSi1
    return result


def decisionTree(trainData, trainDataResult, edgeOfTree,
                 numberOfFeature=1, maximumDepth=10, minPowerOfSelection=1, minEntropy=0.05):
    # depth- how deep tree can be
    # minCountOfSelection- if selection lower than this value- we make leaf
    # minEntropy- if Entropy we gain after split the selection lower than this value- we make leaf
    # numberOfFeature- how much Feature selection function will return to us parameters
    entropy = getEntropy(len(trainData), trainDataResult)
    print(entropy)
    if edgeOfTree.depth > maximumDepth or len(trainData) < minPowerOfSelection or entropy < minEntropy:
        edgeOfTree.leaf = True
        edgeOfTree.leafLabel = np.bincount(trainDataResult) / len(trainDataResult)
        return
    countOfCharacteristic = len(trainData[0])
    lossForCharacteristic = []
    tauForCharacteristic = []
    for indexOfCharacteristic in range(countOfCharacteristic):
        xs = trainData[:, indexOfCharacteristic]
        allTau = np.arange(min(xs), max(xs), 0.1)  # all possible threshold values
        w, v = np.meshgrid(xs, allTau)
        arrayOfDirections = w > v  # direction left or right son
        # i element is how much of selection goes to the right son with allTau[i]
        # countOfRightElements = np.count_nonzero(arrayOfDirections, axis=1)
        allPossibleValuesLossFunctions = np.apply_along_axis(lossFunction, 1, arrayOfDirections, trainDataResult) # apply to each cols not to each element
        indexOfMinLoss = np.argmin(allPossibleValuesLossFunctions)  # index of tau with minimum loss
        lossForCharacteristic.append(np.min(allPossibleValuesLossFunctions))
        tauForCharacteristic.append(allTau[indexOfMinLoss])
    indexOfSearchCharacteristic = np.argmin(lossForCharacteristic)
    resultTau = tauForCharacteristic[indexOfSearchCharacteristic]
    edgeOfTree.tau = resultTau
    edgeOfTree.indexOfSearchCharacteristic = indexOfSearchCharacteristic

    edgeOfTree.createLeftSon()
    decisionTree(trainData[trainData[:, indexOfCharacteristic] > resultTau], trainDataResult[np.nonzero(trainData > resultTau)[0]],
                 edgeOfTree.getLeftSon(), numberOfFeature, maximumDepth, minPowerOfSelection, minEntropy)

    edgeOfTree.createRightSon()
    decisionTree(trainData[trainData[:, indexOfCharacteristic] < resultTau], trainDataResult[np.nonzero(trainData < resultTau)[0]],
                 edgeOfTree.getRightSon(), numberOfFeature, maximumDepth, minPowerOfSelection, minEntropy)
    return 0


def trainDecisionTreeClassification(trainData, trainDataResult,
                                    numberOfFeature=1, maximumDepth=10, minPowerOfSelection=1, minEntropy=0.05):
    root = Tree()
    decisionTree(trainData, trainDataResult, root, numberOfFeature, maximumDepth, minPowerOfSelection, minEntropy)
    return root


trainGenData = np.array([[np.random.random()*10], [np.random.random()*10],
                         [np.random.random()*10], [np.random.random()*10],
                         [np.random.random()*10], [np.random.random()*10],
                         [np.random.random()*10], [np.random.random()*10],
                         [np.random.random()*10], [np.random.random()*10]])
trainGenDataResult = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
model = trainDecisionTreeClassification(trainGenData, trainGenDataResult)
model.evaluate(trainGenData[0])


# TODO print tree