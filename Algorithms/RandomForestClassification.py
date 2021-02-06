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

    entropyOfSi0 = getEntropy(Ni, classesForEachX[arrOfDirForEachX == False])
    entropyOfSi1 = getEntropy(Ni, classesForEachX[arrOfDirForEachX])
    result = Ni0 / Ni * entropyOfSi0 + Ni1 / Ni * entropyOfSi1
    return result


def randomDecisionTree(trainData, trainDataResult, edgeOfTree, countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minEntropy):
    # depth- how deep tree can be
    # minCountOfSelection- if selection lower than this value- we make leaf
    # minEntropy- if Entropy we gain after split the selection lower than this value- we make leaf
    # koefOfRandom- how much our tree will be changed by random node optimization
    entropy = getEntropy(len(trainData), trainDataResult)
    # print(round(entropy, 2))
    isOneFeature = True
    countOfCharacteristic = len(trainData[0])

    for i in range(countOfCharacteristic):
        if len(set(trainData[:, i])) != 1:
            isOneFeature = False
    if edgeOfTree.depth > maximumDepth or len(trainData) <= minPowerOfSelection or entropy < minEntropy or isOneFeature:
        edgeOfTree.leaf = True
        t = np.bincount(trainDataResult, minlength=countOfClasses)
        edgeOfTree.leafLabel = t / len(trainDataResult)
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
        t = np.bincount(trainDataResult, minlength=countOfClasses)
        edgeOfTree.leafLabel = t / len(trainDataResult)
        return
    edgeOfTree.tau = resultTau
    edgeOfTree.indexOfSearchFeature = indexOfFeature

    edgeOfTree.createLeftSon()
    randomDecisionTree(trainData[trainData[:, indexOfFeature] < resultTau],
                       trainDataResult[np.nonzero(trainData[:, indexOfFeature] < resultTau)[0]],
                       edgeOfTree.getLeftSon(), countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minEntropy)

    edgeOfTree.createRightSon()
    randomDecisionTree(trainData[trainData[:, indexOfFeature] >= resultTau],
                       trainDataResult[np.nonzero(trainData[:, indexOfFeature] >= resultTau)[0]],
                       edgeOfTree.getRightSon(), countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minEntropy)
    return edgeOfTree


def trainRandomDecisionTreeClassification(trainData, trainDataResult, koefOfRandom,
                                          maximumDepth=10, minPowerOfSelection=20, minEntropy=0.05):
    root = Tree()
    countOfClasses = len(set(trainDataResult))
    randomDecisionTree(trainData, trainDataResult, root, countOfClasses, koefOfRandom,
                       maximumDepth, minPowerOfSelection, minEntropy)
    return root


def trainRandomForestClassifier(trainData, trainDataResult, koefOfRandom, countOfTrees,
                                maximumDepth=10, minPowerOfSelection=20, minEntropy=0.05):
    forest = RandomForest(countOfTrees)
    for i in range(countOfTrees):
        root = trainRandomDecisionTreeClassification(trainData, trainDataResult, koefOfRandom,
                                                               maximumDepth, minPowerOfSelection, minEntropy)
        forest.setTree(i, root)
    return forest

