import numpy as np
import pandas as pd
from Algorithms.DecisionTreeClassification import trainDecisionTreeClassification
from Algorithms.Metrics import getConfusionMatrix


trainDataFrame = pd.read_csv("../Datasets/phones parametres/train.csv")
print(trainDataFrame.head())
namesOfCol = trainDataFrame.columns
trainData = trainDataFrame.to_numpy()[:, :-1]
trainDataResult = np.int32(trainDataFrame.to_numpy()[:, -1])

N = 1200
model = trainDecisionTreeClassification(trainData[:N], trainDataResult[:N], stepThreshold=0.1)
resultsOfAlg = [np.argmax(model.evaluate(x)) for x in trainData[N:]]
print(getConfusionMatrix(resultsOfAlg, trainDataResult[N:]))
