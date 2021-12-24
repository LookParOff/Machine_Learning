import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import time


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


def getInformationOfBooks(titles):
    dopInfOfBooks = pd.read_csv("../Datasets/book rate rec/BX_Books.csv", sep=";")
    df1 = dopInfOfBooks[dopInfOfBooks["Book-Title"].isin(titles)][
        ['ISBN', "Book-Title", 'Book-Author', 'Image-URL-M', 'Image-URL-S']]
    df2 = df1[~df1["Book-Title"].duplicated()]
    return df2


def correlationRecommendation(ratingDf, title):
    book = ratingDf.loc[:, title]
    recs = ratingDf.corrwith(book).sort_values(ascending=False)
    print(recs[:15].index)
    return recs


def LFRecommendation():  # Latent Factor Model https://youtu.be/J-QueLndVI8?t=2648
    pass


def kNeighbourRecommendation():
    pass


if __name__ == "__main__":
    t = time()
    # users = workWithUsers()
    booksRate, booksDescribe = workWithBook()
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
    mean_rate = np.nanmean(matrixOfRating)
    matrixOfRating.replace(np.nan, mean_rate, inplace=True)
    matrixOfRating = matrixOfRating - mean_rate
    print(f"Pivot matrix is done in {time() - t} secs")

    t = time()
    # union the duplicate books
    matrixOfRating = matrixOfRating.groupby(by=matrixOfRating.columns, axis=1).sum()
    print(f"Duplicates is gone in {time() - t} secs")
    t = time()
    popularBooks = matrixOfRating.sum().sort_values(ascending=False).index
    # (36582, 11173)
    # maybe should give a new name
    # matrixOfRating:
    #           title title2 ...
    # User-ID     6     2
    # 10         Nan    0
    # 12          10    2
    # ...
    print(matrixOfRating)
    print("harry potter 1:")
    recs1 = correlationRecommendation(matrixOfRating, "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))")
    print("Lord of Chaos:")
    recs2 = correlationRecommendation(matrixOfRating, "Lord of Chaos (The Wheel of Time, Book 6)")
