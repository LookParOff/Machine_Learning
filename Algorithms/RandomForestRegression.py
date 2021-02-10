import numpy as np


class Tree:
    def __init__(self):
        self.tau = None
        self.indexOfSearchFeature = None
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
        self.leftSon.depth = self.depth + 1

    def createRightSon(self):
        self.rightSon = Tree()
        self.rightSon.parent = self
        self.rightSon.depth = self.depth + 1

    def getLeftSon(self):
        return self.leftSon

    def getRightSon(self):
        return self.rightSon

    def __str__(self):
        printArray[self.depth].append((self.tau, self.indexOfSearchFeature))
        self.leftSon.__str__()
        self.rightSon.__str__()


class RandomForest:
    def __init__(self, countOfTrees):
        self.countOfTrees = countOfTrees
        self.listOfTrees = [Tree() for _ in range(countOfTrees)]

    def setTree(self, i, rootEdge):
        self.listOfTrees[i] = rootEdge

    def evaluate(self, x):
        sumOfAllAnswers = np.array(self.listOfTrees[0].evaluate(x)*0)
        for i in range(self.countOfTrees):
            sumOfAllAnswers += self.listOfTrees[i].evaluate(x)
        return sumOfAllAnswers / self.countOfTrees


printArray = [[] for _ in range(100)]


def lossFunction(arrOfDirForEachX, valueOfRegressions):
    # arrOfDir- True or False for each element of selection, True or False means we go to the left or right son
    # Ni- len of current selection, which went to this edge
    stdOnLeftSon = np.std(valueOfRegressions[arrOfDirForEachX == False])
    stdOnRightSon = np.std(valueOfRegressions[arrOfDirForEachX])
    result = (stdOnLeftSon + stdOnRightSon) / 2
    return result


def randomDecisionTree(trainData, trainDataResult, edgeOfTree, countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minDeviation):
    # depth- how deep tree can be
    # minCountOfSelection- if selection lower than this value- we make leaf
    # minDeviation- if standart deviation we gain after split the selection lower than this value- we make leaf
    # koefOfRandom- how much our tree will be changed by random node optimization
    isOneFeature = True
    countOfCharacteristic = len(trainData[0])
    deviation = np.std(trainDataResult)

    for i in range(countOfCharacteristic):
        if len(set(trainData[:, i])) != 1:
            isOneFeature = False
    if edgeOfTree.depth > maximumDepth or len(trainData) <= minPowerOfSelection or deviation < minDeviation or isOneFeature:
        edgeOfTree.leaf = True
        t = np.mean(trainDataResult)
        edgeOfTree.leafLabel = t
        return

    randomNodeFeatures = []
    randomNodeThreshold = []
    for _ in range(koefOfRandom):
        randomNodeFeatures.append(np.random.randint(0, countOfCharacteristic))
        features = trainData[:, randomNodeFeatures[-1]]
        randomNodeThreshold.append(np.random.uniform(min(features), max(features)))
    minLossFunc = 2**32
    indexOfFeature = None
    resultTau = None
    for feature, thresholdValue in zip(randomNodeFeatures, randomNodeThreshold):
        xs = trainData[:, feature]
        if len(set(xs)) == 1:
            continue
        arrayOfDirections = xs >= thresholdValue  # direction left or right son, False or True
        valueOfLossFunc = lossFunction(arrayOfDirections, trainDataResult)
        if valueOfLossFunc < minLossFunc:
            minLossFunc = valueOfLossFunc
            indexOfFeature = feature
            resultTau = thresholdValue
    if resultTau is None:
        edgeOfTree.leaf = True
        t = np.mean(trainDataResult)
        edgeOfTree.leafLabel = t
        return
    edgeOfTree.tau = resultTau
    edgeOfTree.indexOfSearchFeature = indexOfFeature

    edgeOfTree.createLeftSon()
    randomDecisionTree(trainData[trainData[:, indexOfFeature] < resultTau],
                       trainDataResult[np.nonzero(trainData[:, indexOfFeature] < resultTau)[0]],
                       edgeOfTree.getLeftSon(), countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minDeviation)

    edgeOfTree.createRightSon()
    randomDecisionTree(trainData[trainData[:, indexOfFeature] >= resultTau],
                       trainDataResult[np.nonzero(trainData[:, indexOfFeature] >= resultTau)[0]],
                       edgeOfTree.getRightSon(), countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minDeviation)
    return edgeOfTree


def trainRandomDecisionTreeRegression(trainData, trainDataResult, koefOfRandom,
                                      maximumDepth=10, minPowerOfSelection=20, minDeviation=0.05):
    root = Tree()
    countOfClasses = len(set(trainDataResult))
    randomDecisionTree(trainData, trainDataResult, root, countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minDeviation)
    return root


def trainRandomForestRegression(trainData, trainDataResult, koefOfRandom, countOfTrees,
                                maximumDepth=10, minPowerOfSelection=20, minDeviation=0.05):
    forest = RandomForest(countOfTrees)
    for i in range(countOfTrees):
        root = trainRandomDecisionTreeRegression(trainData, trainDataResult, koefOfRandom,
                                                 maximumDepth, minPowerOfSelection, minDeviation)
        forest.setTree(i, root)
    return forest
