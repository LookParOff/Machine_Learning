import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from time import time
# https://proglib.io/p/sovremennye-rekomendatelnye-sistemy-2021-03-02


class AutoEncoder(torch.nn.Module):
    def __init__(self, countOfInpNeurons):
        super().__init__()
        self.relu = torch.nn.ReLU()

        self.encode1 = torch.nn.Linear(countOfInpNeurons, 1000)
        self.encode2 = torch.nn.Linear(1000, 750)
        self.encode3 = torch.nn.Linear(750, 500)

        self.decode1 = torch.nn.Linear(500, 750)
        self.decode2 = torch.nn.Linear(750, 1000)
        self.decode3 = torch.nn.Linear(1000, countOfInpNeurons)

    def encode(self, x: torch.Tensor):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        return x

    def decode(self, x: torch.Tensor):
        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.decode(x)
        return x


def getAutoEncoder(countOfInpNeurons):
    autoEncoder = AutoEncoder(countOfInpNeurons)
    autoEncoder.to(device=device)
    return autoEncoder


class RecommendationNetwork(torch.nn.Module):
    def __init__(self, countOfInputNeurons):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, countOfInputNeurons)
        self.linear1 = torch.nn.Linear(32, 250)
        self.linear2 = torch.nn.Linear(250, 250)
        # self.linear3 = torch.nn.Linear(250, 250)
        self.linear4 = torch.nn.Linear(250, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        x = self.linear4(x)
        return x


def getRecNet(countOfInpNeurons):
    net = RecommendationNetwork(countOfInpNeurons)
    net.to(device=device)
    return net


def getPivotTable() -> pd.DataFrame:
    t = time()
    rawBooks = pd.read_csv("../Datasets/book rate rec/BX-Book-Ratings.csv", sep=";")
    rawBooks = rawBooks[rawBooks["Book-Rating"] > 0]  # drop the useless zeros and nan
    countOfEveryBook = pd.value_counts(rawBooks["ISBN"])
    countOfEveryBook = countOfEveryBook[countOfEveryBook > 25].index
    booksRate = rawBooks[rawBooks["ISBN"].isin(countOfEveryBook)]  # filter, so books at least 10 people read

    booksDescribe = pd.read_csv("../Datasets/book rate rec/BX_Books.csv", sep=";")
    booksDescribe = booksDescribe[["ISBN", "Book-Title"]]
    booksDescribe = booksDescribe[booksDescribe["ISBN"].isin(booksRate["ISBN"])]

    booksRate, booksDescribe = booksRate[:], booksDescribe[:]
    # now we make matrix with rowIndex is User-ID and colIndex is ISBN. On intersection- rate
    print(f"Preprocess of data is over in {time() - t}secs")

    t = time()
    matrixOfRating = pd.pivot_table(booksRate, values="Book-Rating", index="User-ID", columns="ISBN")
    # print("mean", np.nanmean(booksRate["Book-Rating"]), "median", np.nanmedian(booksRate["Book-Rating"]))
    # instead of isbn there are will be the title of books:
    newColumns = []
    for col in matrixOfRating.columns:
        newColumns.append(col)
    newColumns = pd.Series(newColumns).replace(booksDescribe["ISBN"].to_list(), booksDescribe["Book-Title"].to_list())
    matrixOfRating.columns = newColumns
    # keep only books with proper names:
    matrixOfRating = matrixOfRating.loc[:, matrixOfRating.columns.isin(booksDescribe["Book-Title"])]
    # make statistic magic:
    # mean_rate = np.nanmean(matrixOfRating)
    # matrixOfRating.replace(np.nan, np.nanmean(matrixOfRating), inplace=True)  # fillna
    # matrixOfRating = matrixOfRating - np.nanmean(matrixOfRating)
    print(f"Pivot matrix is done in {time() - t} secs")
    t = time()
    # union the duplicate books
    matrixOfRating = matrixOfRating.groupby(by=matrixOfRating.columns, axis=1).max()
    # matrixOfRating.fillna(0, inplace=True)
    print(f"Duplicates is gone in {time() - t} secs")
    return matrixOfRating


def getAccuracy(algorithmValues, correctValues):
    # threshold = np.mean([np.median(correctValues), np.max(correctValues)])  # third quartile
    equalValues = 0
    wrongValues = 0
    for algBatch, corrBatch in zip(algorithmValues, correctValues):
        alg = algBatch.cpu().detach().numpy()
        corr = corrBatch.cpu().detach().numpy()
        equalValues += np.count_nonzero(np.abs(alg - corr) < 0.3)
        wrongValues += np.count_nonzero(np.abs(alg - corr) >= 0.3)
    return equalValues / (equalValues + wrongValues)


def getPrecisionRecall(algorithmValues, correctValues):
    TP = 0  # Recommended relevant
    TN = 0  # Didn't recommended irrelevant
    FP = 0  # Recommended irrelevant  (true_y=0 & pred_y=1)
    FN = 0  # Didn't recommended relevant
    # threshold = np.mean([np.median(correctValues), np.max(correctValues)])  # third quartile
    threshold = torch.mean(torch.tensor([torch.median(correctValues), torch.max(correctValues)]))  # third quartile
    # threshold = 0.7
    # print(threshold)
    for alg, corr in zip(algorithmValues, correctValues):
        alg, corr = alg[0], corr[0]
        if alg >= threshold and corr >= threshold:
            TP += 1
        elif alg <= threshold and corr <= threshold:
            TN += 1
        elif alg > threshold and corr < threshold:
            FP += 1
        elif alg < threshold and corr > threshold:
            FN += 1
    try:
        # probably in this task we need take care about precision, rather recall
        precision = TP / (TP + FP)  # which part algorithm recognize as 1 class, and it's correct
        recall = TP / (TP + FN)  # which part algorithm recognize as 1 class of all elements of 1 class
    except ZeroDivisionError:
        print(0, -1)
        precision, recall = 0, -1
    return precision, recall


def getRecommendation(trainedModel, userId, matrixR: pd.DataFrame, pcaEmbeding):
    """
    trainedModel is a function, which take number of row in matrixOfRating of userId and
    number of column in matrixOfRating of bookTitle.
    For example for userId 82825 it will be 12938 and
    for bookTitle Harry Potter and the Chamber of Secrets (Book 2) it will be 2087.
    So trainedModel(12938, 2087) will return rating of user 82825 on HP (10)

    :returns list of the most rating books
    """
    ratings = []
    userIndex = np.where(matrixR.index == userId)[0][0]
    for bookIndex in range(matrixR.shape[1]):
        inp = np.concatenate([pcaEmbeding[0][userIndex], pcaEmbeding[2][bookIndex]])
        inp = torch.tensor(inp, device=device)
        ratings.append(trainedModel(inp)[0].cpu().detach().numpy())
    ratings = np.asarray(ratings)
    sortIndexes = np.argsort(ratings)[::-1]
    ratings = ratings[sortIndexes]
    recommendedTitles = matrixR.columns[sortIndexes]

    # threshold = np.mean([np.median(ratings), np.max(ratings)])  # third quartile
    # recommendedTitles = recommendedTitles[ratings > threshold]
    # ratings = ratings[ratings > threshold]
    # recs[0]["Harry" in recs[0]]
    return recommendedTitles[:10], ratings[:10]


# def getDataLoaders(pivot_table: np.array):
#     percentOfSplit = 0.15
#     trainTable = pivot_table[:pivot_table.shape[0] - int(pivot_table.shape[0] * percentOfSplit), :]
#     validTable = pivot_table[pivot_table.shape[0] - int(pivot_table.shape[0] * percentOfSplit):, :]
#
#     trainTensor = torch.tensor(trainTable, dtype=torch.float32, device=device)
#     ValidTensor = torch.tensor(validTable, dtype=torch.float32, device=device)
#
#     trainData = TensorDataset(trainTensor, trainTensor)
#     validData = TensorDataset(ValidTensor, ValidTensor)
#     trainDataLoader = DataLoader(trainData, batch_size=64)
#     validDataLoader = DataLoader(validData, batch_size=validTable.shape[0])
#     return trainDataLoader, validDataLoader


def getDataSets(pivotTable) -> (np.array, np.array):
    indexesUsers, indexesItems = np.where(pivotTable != 0)
    inputData = []
    outputData = []
    for user, item in zip(indexesUsers, indexesItems):
        inp = np.concatenate([pivotTable[user, :], pivotTable[:, item]])  # input for neural net
        out = pivotTable[user, item]  # what we need it return (rate maybe)
        inputData.append(inp)
        outputData.append(out)
    inputData = np.asarray(inputData)
    outputData = np.asarray(outputData).reshape((len(outputData), 1))
    print(inputData)
    print(outputData)
    return inputData, outputData


def getDataLoaders(inpData, outData):
    percentOfSplit = 0.15

    trainInpData = inpData[:inpData.shape[0] - int(inpData.shape[0] * percentOfSplit), :]
    trainOutData = outData[:outData.shape[0] - int(outData.shape[0] * percentOfSplit)]
    validInpData = inpData[inpData.shape[0] - int(inpData.shape[0] * percentOfSplit):, :]
    validOutData = outData[outData.shape[0] - int(outData.shape[0] * percentOfSplit):]

    trainInpTensor = torch.tensor(trainInpData, dtype=torch.float32, device=device)
    trainOutTensor = torch.tensor(trainOutData, dtype=torch.float32, device=device)
    validInpTensor = torch.tensor(validInpData, dtype=torch.float32, device=device)
    validOutTensor = torch.tensor(validOutData, dtype=torch.float32, device=device)

    trainData = TensorDataset(trainInpTensor, trainOutTensor)
    validData = TensorDataset(validInpTensor, validOutTensor)

    trainDataLoader = DataLoader(trainData, batch_size=64)
    validDataLoader = DataLoader(validData, batch_size=validOutData.shape[0])
    return trainDataLoader, validDataLoader


def fit(model, epochs, loss_func, opt, trainDL, validDL):
    # metric = torchmetrics.ExplainedVariance()
    # metric.to(device)
    for epoch in range(epochs):
        for x, y in trainDL:
            # Forward pass: compute predicted y by passing x to the model.
            yPredict = model(x)
            # smth = torch.Tensor(yPredict.size(0)).cuda().fill_(1.0)
            loss = loss_func(yPredict, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch < 6:
            for x, y in validDL:
                # single iteration
                yPredict = model(x)
                # metric(yPredict, y)
            print(f"On epoch {epoch}:\n"
                  f"\tloss = {loss.item()}\n"
                  f"\taccuracy = {getAccuracy(yPredict, y)}\n"
                  f"\tprec and rec = {getPrecisionRecall(yPredict, y)}")
    for x, y in validDL:
        # single iteration
        yPredict = model(x)
        # metric(yPredict, y)
    print(f"In the end:\n"
          f"\tloss = {loss.item()}\n"
          f"\taccuracy = {getAccuracy(yPredict, y)}\n"
          f"\tprec and rec = {getPrecisionRecall(yPredict, y)}")
    return model


# TODO 1) AutoEncoder to create latent vectors for users and items
# TODO 2) create NN. input: latent user and latent item. output: one digit of rating, or bool recommend or not
if __name__ == "__main__":
    # todo normalizing of each user and each item
    device = torch.device("cuda:0")
    pivTabDF = getPivotTable()  # df

    pivTab = np.asarray(pivTabDF)
    # meanValuesUsers = np.nanmean(pivTab, axis=1)
    # meanValuesItems = np.nanmean(pivTab, axis=0)
    # meanValuesUsers = np.reshape(meanValuesUsers, (meanValuesUsers.shape[0], 1))
    # meanValuesUsers = np.repeat(meanValuesUsers, pivTab.shape[1], axis=1)
    # meanValuesItems = np.array([meanValuesItems for _ in range(pivTab.shape[0])])
    # pivTab -= (meanValuesUsers + meanValuesItems)
    # pivTab -= meanValuesItems
    # pivTab = np.nan_to_num(pivTab)

    # pivTabDF = pivTabDF - pivTabDF.mean(axis=1, skipna=True)
    # pivTabDF = pivTabDF - pivTabDF.mean(axis=0, skipna=True)
    # mean = pivTabDF.mean(skipna=True)
    # std = pivTabDF.mean(skipna=True)
    # pivTabDF = (pivTabDF - mean) / std
    # pivTab = pivTabDF.fillna(0)
    # pivTab = np.asarray(pivTab)

    # pcaPivTab = torch.pca_lowrank(torch.Tensor(pivTab))
    # pcaPivTab = (pcaPivTab[0].numpy(), pcaPivTab[1].numpy(), pcaPivTab[2].numpy())
    inputDataSet, outputDataSet = getDataSets(pivTab)
    trainDL, validDL = getDataLoaders(inputDataSet, outputDataSet)
    emptyModel = getRecNet(inputDataSet[0].shape[0])

    # loss_fn = torch.nn.CosineEmbeddingLoss(reduction='sum')  # was sum
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 0.001
    epochs = 10
    optimizer = torch.optim.Adam(emptyModel.parameters(), lr=learning_rate)
    trainedModel = fit(emptyModel, epochs, loss_fn, optimizer, trainDL, validDL)
    # print(getRecommendation(trainedModel, 82825, pivTabDF, pcaPivTab))
    # validNP = validDL.dataset.tensors[0].cpu().detach().numpy()
    # enc = trainedModel(validDL.dataset.tensors[0])
    # encNP = enc.cpu().detach().numpy()
    # print(np.count_nonzero(np.abs(encNP.round() - validNP.round()) >= 0.5))  # wrong values
    # print(np.count_nonzero(validNP))

    # validNP = validDL.dataset.tensors[0].cpu().detach().numpy()
    # enc = trainedModel(validDL.dataset.tensors[0])
    # encNP = enc.cpu().detach().numpy()
    #
    # validNP = (validNP * std) + mean
    # encNP = (encNP * std) + mean
    #
    # print(np.count_nonzero(np.abs(encNP.round() - validNP.round()) >= 0.5))  # wrong values
    # print(np.count_nonzero(validNP))
