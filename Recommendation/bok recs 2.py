import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import time


# pd.set_option('display.max_columns', 6)


np.seterr(all='raise')


def workWithUsers() -> pd.DataFrame:
    raw_users = pd.read_csv("../Datasets/book rate rec/BX-Users.csv", sep=";")
    # raw_users = raw_users.dropna()
    location = raw_users["Location"]
    country = []
    for place in location:
        country.append(place.split(", ")[-1])
    # print(pd.value_counts(country))

    # d = {"User-ID": raw_users["User-ID"].dropna(), "Country": country, "Age": raw_users.dropna()["Age"]}
    d = {"User-ID": raw_users["User-ID"], "Country": country, "Age": raw_users["Age"]}
    new_users = pd.DataFrame(data=d)
    return new_users


def workWithBook() -> (pd.DataFrame, pd.DataFrame):
    rawBooks = pd.read_csv("../Datasets/book rate rec/BX-Book-Ratings.csv", sep=";")
    # rawBooks = rawBooks[rawBooks["User-ID"].isin(users["User-ID"])]
    rawBooks = rawBooks[rawBooks["Book-Rating"] > 0]  # drop the useless zeros and nan
    countOfEveryBook = pd.value_counts(rawBooks["ISBN"])
    countOfEveryBook = countOfEveryBook[countOfEveryBook > 7].index
    bRate = rawBooks[rawBooks["ISBN"].isin(countOfEveryBook)]  # filter, so books at least 10 people read

    bDescribe = pd.read_csv("../Datasets/book rate rec/BX_Books.csv", sep=";")
    bDescribe = bDescribe[["ISBN", "Book-Title"]]
    bDescribe = bDescribe[bDescribe["ISBN"].isin(bRate["ISBN"])]
    return bRate, bDescribe


def getInformationOfBooks(matOfRate):
    """
    :return: dataFrame, which contain book information and book's order based on popularity
    """
    popularBooks = matOfRate.sum().sort_values(ascending=False)
    popularBooks = pd.DataFrame(popularBooks)
    popularBooks.reset_index(level=0, inplace=True)
    popularBooks.columns = ["Book-Title", "Sum-Rate"]

    dopInfOfBooks = pd.read_csv("../Datasets/book rate rec/BX_Books.csv", sep=";")
    df1 = dopInfOfBooks[dopInfOfBooks["Book-Title"].isin(popularBooks["Book-Title"])][
        ['ISBN', "Book-Title", 'Book-Author', 'Image-URL-M', 'Image-URL-S']]
    df2 = df1[~df1["Book-Title"].duplicated()]
    df3 = df2.merge(popularBooks)
    df3.sort_values(by="Sum-Rate", inplace=True, ascending=False)
    df3.drop("Sum-Rate", axis=1, inplace=True)
    # df3.to_json("D:\\output.json", orient="table")
    return df3


def correlationRecommendation(ratingDf, title):
    book = ratingDf.loc[:, title]
    recs = ratingDf.corrwith(book).sort_values(ascending=False)
    print(recs[:15].index)
    return recs


def error1(matrix, numRows, numCols, P, Q, rk=0.1):
    e = 0
    for u, i in zip(numRows, numCols):
        rui = matrix[u][i]
        if rui > 0:  # ?!
            e = e + pow(rui - np.dot(P[u, :], Q[:, i]), 2) + \
                rk * (pow(np.linalg.norm(P[u, :]), 2) + pow(np.linalg.norm(Q[:, i]), 2))
    return e


def error2(matrix, meanValuesUsers, meanValuesItems, numRows, numCols, P, Q):
    e = 0
    for u, i in zip(numRows, numCols):
        rui = matrix[u][i]
        if not np.isnan(rui):
            e += pow(rui - meanValuesUsers[u] - meanValuesItems[i] - np.dot(P[u, :], Q[:, i]), 2)
    return e


def getPrecisionAndRecall(algorithmValues, correctValues):
    TP = 0  # Recommended relevant
    TN = 0  # Didn't recommended irrelevant
    FP = 0  # Recommended irrelevant  (true_y=0 & pred_y=1)
    FN = 0  # Didn't recommended relevant
    threshold = np.mean([np.median(correctValues), np.max(correctValues)])  # third quartile
    for alg, corr in zip(algorithmValues, correctValues):
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
        accuracy = (TP + TN) / len(correctValues)
    except ZeroDivisionError:
        print(0, -1)
        accuracy, precision, recall = 0, 0, -1
    return accuracy, precision, recall


def trainLFM(matrix: np.array, meanValuesUsers, meanValuesItems, steps, lr):
    # Latent Factor Model https://youtu.be/J-QueLndVI8?t=2648
    # matrix- pivot matrix with rate on ij place. i- users and j- book
    # lr- learning rate
    # rk- regularization koef
    st = time()
    K = 10  # features
    P = np.random.random((matrix.shape[0], K))  # latent users
    Q = np.random.random((K, matrix.shape[1]))  # latent books

    numRows, numCols = np.where(~np.isnan(matrix))  # indexes of rows/cols, where values are not nan
    # make validation set:
    indexes = [i for i in range(len(numRows))]
    np.random.shuffle(indexes)
    numRows, numCols = numRows[indexes], numCols[indexes]

    # take 15% for validation:
    validRows, validCols = numRows[:int(len(numRows) * 0.15)], numCols[:int(len(numCols) * 0.15)]
    numRows, numCols = numRows[int(len(numRows) * 0.15):], numCols[int(len(numCols) * 0.15):]
    correctValidValues = []
    for r, c in zip(validRows, validCols):
        correctValidValues.append(matrix[r, c])
        matrix[r, c] = np.nan
    algorithmValues = []
    for r, c in zip(validRows, validCols):
        algorithmValues.append(np.dot(P[r, :], Q[:, c]) + meanValuesUsers[r] + meanValuesItems[c])

    rmse = np.sqrt(error2(matrix, meanValuesUsers, meanValuesItems, numRows, numCols, P, Q) / len(matrix))
    accuracy, precision, recall = getPrecisionAndRecall(algorithmValues, correctValidValues)
    print(f"Without education: \n\tRMSE = {round(rmse, 2)}")
    print(f"\taccuracy = {round(accuracy, 2)}\n"
          f"\tprecision = {round(precision, 2)}; recall = {round(recall, 2)};\n")

    check = 5  # determinate how often we check update of rmse
    indexes = [i for i in range(len(numRows))]

    # training:
    for step in range(steps):
        itera = 0
        np.random.shuffle(indexes)
        numRows, numCols = numRows[indexes], numCols[indexes]
        for u, i in zip(numRows, numCols):
            itera += 1
            rui = matrix[u][i]
            if rui > 0:  # ?!
                # TODO consider about weights
                # TODO regularisation
                #
                # https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b
                # https://towardsdatascience.com/evaluating-recommender-systems-root-means-squared-error-or-mean-absolute-error-1744abc2beac
                eui = rui - meanValuesUsers[u] - meanValuesItems[i] - np.dot(P[u, :], Q[:, i])
                P[u, :] = P[u, :] + lr * eui * Q[:, i]
                Q[:, i] = Q[:, i] + lr * eui * P[u, :]

        if step % check == 0:
            print(f"Step {step}:")
            pastRmse = rmse
            rmse = np.sqrt(error2(matrix, meanValuesUsers, meanValuesItems, numRows, numCols, P, Q) / len(matrix))
            print(f"\tRSME = {round(rmse, 3)}."
                  f" Shift per step = {round((pastRmse - rmse) / check, 4)}")
            if (pastRmse - rmse) / check < 0.003:  # next iteration is almost useless
                break

            # metrics on validation selection
            algorithmValues = []
            for r, c in zip(validRows, validCols):
                algorithmValues.append(np.dot(P[r, :], Q[:, c]) + meanValuesUsers[r] + meanValuesItems[c])
            pastPrecision = precision
            accuracy, precision, recall = getPrecisionAndRecall(algorithmValues, correctValidValues)
            print("\tmean value of algValues =", round(np.mean(algorithmValues), 2))
            print(f"\taccuracy = {round(accuracy, 2)}\n"
                  f"\tprecision = {round(precision, 2)}; recall = {round(recall, 2)};\n")
            if precision < pastPrecision:
                break  # precision start getting bigger. That's means overfitting
    print(f"Train is over in {round(time() - st)} secs")
    return lambda us, it: np.dot(P[us, :], Q[:, it])


def getRecommendation(trainedModel, userId, matrixR: pd.DataFrame, meanOfUser, meanOfBooks):
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
        ratings.append(trainedModel(userIndex, bookIndex) + meanOfUser + meanOfBooks[bookIndex])
    ratings = np.asarray(ratings)
    sortIndexes = np.argsort(ratings)[::-1]
    ratings = ratings[sortIndexes]
    recommendedTitles = matrixR.columns[sortIndexes]

    threshold = np.mean([np.median(ratings), np.max(ratings)])  # third quartile
    recommendedTitles = recommendedTitles[ratings > threshold]
    ratings = ratings[ratings > threshold]
    return recommendedTitles, ratings


if __name__ == "__main__":
    t = time()
    # users = workWithUsers()
    booksRateALL, booksDescribeALL = workWithBook()
    booksRate, booksDescribe = booksRateALL[:], booksDescribeALL[:]
    # users = users[users["User-ID"].isin(booksRate["User-ID"])]
    # now we make matrix with rowIndex is User-ID and colIndex is ISBN. On intersection- rate
    print(f"Preprocess of data is over in {time() - t}secs")

    t = time()
    matrixOfRating = pd.pivot_table(booksRate, values="Book-Rating", index="User-ID", columns="ISBN")
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

    t = time()
    # matrixOfRating:
    #           title title2 ...
    # User-ID     6     2
    # 10         Nan    0
    # 12          10    2
    # ...
    # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    # Lord of Chaos (The Wheel of Time, Book 6)

    matNp = np.asarray(matrixOfRating)
    allRows, allCols = np.where(~np.isnan(matNp))  # indexes of rows/cols, where values are not nan
    indexes = [i for i in range(len(allRows))]
    np.random.shuffle(indexes)
    allRows, allCols = allRows[indexes], allCols[indexes]
    testRows, testCols = allRows[:int(len(allRows) * 0.15)], allCols[:int(len(allCols) * 0.15)]  # take 15% for testing
    corrValues = []
    for r, c in zip(testRows, testCols):
        corrValues.append(matNp[r, c])
        matNp[r, c] = np.nan
    # mean values for users and item. Made it for optimisation purpose
    meanValuesUsers = np.nanmean(matNp, axis=1)
    meanValuesItems = np.nanmean(matNp, axis=0)
    model = trainLFM(matNp, meanValuesUsers, meanValuesItems, steps=100, lr=0.01)
    algValues = []
    for r, c in zip(testRows, testCols):
        algValues.append(model(r, c) + meanValuesUsers[r] + meanValuesItems[c])

    acc, pr, rec = getPrecisionAndRecall(algValues, corrValues)
    print(f"accuracy = {round(acc, 2)}\n"
          f"precision = {round(pr, 2)}; recall = {round(rec, 2)};\n")
    # todo add test users. They read only Harry Potter without one book
    recs = getRecommendation(model, 82825, matrixOfRating,
                             meanValuesUsers[np.where(matrixOfRating.index == 82825)[0][0]], meanValuesItems)
