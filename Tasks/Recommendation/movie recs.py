import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import time


# rate = pd.read_csv("../Datasets/movie dataset collaborative recommendation/ratings.csv", sep=",")
# rate.drop("timestamp", axis=1, inplace=True)
# matrixOfRating = pd.pivot_table(rate, values="rating", index="userId", columns="movieId")


def workWithMovie() -> pd.DataFrame:
    rawMovies = pd.read_csv("../Datasets/huge movie rating dataset/ratings_small.csv", sep=",")
    countOfEveryMovie = pd.value_counts(rawMovies["movieId"])
    countOfEveryMovie = countOfEveryMovie[countOfEveryMovie > 7].index
    mRate = rawMovies[rawMovies["movieId"].isin(countOfEveryMovie)]  # filter, so books at least 7 people read
    return mRate


def getInformationOfMovies(matOfRate):
    """
    :return: dataFrame, which contain information about movie and movie's order based on popularity
    """
    mDescribe = pd.read_csv("../Datasets/huge movie rating dataset/links_small.csv", sep=",")
    mDescribe = mDescribe[["imdb_id", "title", "poster_path"]]  # how use poster_path?
    mDescribe = mDescribe[mDescribe["movieId"].isin(matOfRate.columns)]
    mMetaData = pd.read_csv("../Datasets/huge movie rating dataset/movies_metadata.csv", sep=",")
    mMetaData = mMetaData[[]]
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


if __name__ == "__main__":
    t = time()
    # users = workWithUsers()
    moviesRateALL = workWithMovie()
    moviesRate = moviesRateALL[:]
    # users = users[users["User-ID"].isin(booksRate["User-ID"])]
    # now we make matrix with rowIndex is User-ID and colIndex is ISBN. On intersection- rate
    print(f"Preprocess of data is over in {time() - t}secs")

    t = time()
    matrixOfRating = pd.pivot_table(moviesRate, values="rating", index="userId", columns="movieId")
