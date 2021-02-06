import numpy as np
import pandas as pd
from Algorithms.Metrics import getPrecisionAndRecall, getFScore
from Algorithms.DecisionTreeClassification import trainDecisionTreeClassification
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
# print(dfTest)
