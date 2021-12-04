# data- https://www.kaggle.com/ruchi798/bookcrossing-dataset
# 105283 uniq users
# 340556 uniq books
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
SET_OF_ALLOWED_SYMBOLS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "X"}
pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 300)  # or 1000
pd.set_option('display.max_colwidth', 1000)  # or 199


def check_allowed_symb(word):
    for letter in word:
        if letter not in SET_OF_ALLOWED_SYMBOLS:
            return False
    return True


def log_search(arr, num, i, j):
    center = (i + j) // 2
    if arr[center] == num:
        return center
    if j - i > 0:
        if arr[center] < num:
            return log_search(arr, num, center + 1, j)
        elif arr[center] > num:
            return log_search(arr, num, i, center - 1)
    return -1


# drop people, who made low count of rate
df = pd.read_csv("../Datasets/BX-Book-Ratings.csv", sep=";", error_bad_lines=False, header=0)
# for index, row in enumerate(df.iterrows()):
#     for letter in row[1]["ISBN"]:
#         if letter not in SET_OF_ALLOWED_SYMBOLS:
#             df.drop(index=index)


df_corr_user = pd.DataFrame(df.groupby(["User-ID"], ).size())
COUNT_OF_RATED_BOOKS = 50
df_corr_user = df_corr_user[df_corr_user >= COUNT_OF_RATED_BOOKS].dropna()
print(df_corr_user)
print(df_corr_user.index)

# drop book, which no one rate
df_corr_book = pd.DataFrame(df.groupby(["ISBN"], ).size())
df_corr_book = df_corr_book[df_corr_book >= 100].dropna()
print(df_corr_book)
print(df_corr_book.index)
map_of_ISBN_to_index = dict({df_corr_book.index[i]: i for i in range(len(df_corr_book.index))})

# matrix in _i_j place is rate by user_i on book_j
matrix_of_rate = np.zeros((df_corr_user.index.shape[0], df_corr_book.index.shape[0]), dtype="float") + np.nan

iteration = 0
for row in df.itertuples():
    iteration += 1
    if iteration % 50000 == 0:
        print(iteration)
    user_ID = row[1]
    ISBN_of_book = row[2]
    rate = row[3]
    # row of this user in matrix_of_rate
    index_of_user = log_search(df_corr_user.index, user_ID, 0, len(df_corr_user.index) - 1)
    index_of_book = map_of_ISBN_to_index.get(ISBN_of_book)
    if index_of_user != -1 and index_of_book is not None:
        matrix_of_rate[index_of_user][index_of_book] = rate


mean_of_mat = np.nanmean(matrix_of_rate)
matrix_of_rate[np.where(np.isnan(matrix_of_rate))] = mean_of_mat
matrix_of_rate -= mean_of_mat
print("matrix is done")

U, S, Vh = np.linalg.svd(matrix_of_rate)

U_ = U[:, 2]

plt.scatter(U[:6, 0], U[:6, 1])

plt.show()
# TODO мы нашли, что User-ID 183 и 626 достаточно похожи. Посмотрим так ли это
# TODO User-ID 183 и 392- одинаковы

books_of_183 = df[df["User-ID"] == 183]["ISBN"].values
books_of_392 = df[df["User-ID"] == 392]["ISBN"].values
books_of_626 = df[df["User-ID"] == 626]["ISBN"].values
same_books = set()

for el183 in books_of_183:
    for el392 in books_of_392:
        if el183 == el392:
            same_books.add(el392)
print("183 and 392", same_books)

same_books = set()
for el183 in books_of_183:
    for el626 in books_of_626:
        if el183 == el626:
            same_books.add(el626)
print("183 and 626", same_books)
N = 400
plt.scatter(U[:N, 0], U[:N, 1], marker=".")
plt.xlim(-0.001, 0.001)
plt.ylim(-0.004, 0.001)
for i, txt in enumerate(df_corr_user.index[:N]):
    plt.annotate(txt, (U[:N, 0][i], U[:N, 1][i]))
