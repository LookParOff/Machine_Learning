import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

inputFilePath = "..//Datasets/статистика о длинне дорог по регионам.csv"
dfRaw = pd.read_csv(inputFilePath, encoding="windows-1251", sep=";")
dfRaw.replace("-", np.nan, inplace=True)
dfRaw.replace(",", ".", regex=True, inplace=True)
dfRaw.replace("федеральный округ", "ФО", regex=True, inplace=True)
df = pd.DataFrame({"Region": dfRaw[dfRaw.columns[1]],
                   "length": dfRaw[dfRaw.columns[3]].astype(float)})

df = pd.DataFrame(df.groupby(df.columns[0])[df.columns[1]].sum())
# df = df.sort_values("length", ascending=False)
ax = sns.barplot(data=df)
plt.show()
