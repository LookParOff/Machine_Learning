from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
# вдохновение https://medium.com/jovianml/whatsapp-message-exploratory-data-analysis-eda-538560ee1c99
stopWords = set(STOPWORDS)
# useless words:
stopWords.add("я")
stopWords.add("ты")
stopWords.add("как")
stopWords.add("мне")
stopWords.add("с")
stopWords.add("всё")
stopWords.add("так")
stopWords.add("и")
stopWords.add("не")
stopWords.add("в")
stopWords.add("на")
stopWords.add("это")
# system words of VK
stopWords.add("album")
stopWords.add("Прикрепления")
stopWords.add("albumПрикрепления")
stopWords.add("vk")
stopWords.add("jpg")
stopWords.add("http")
stopWords.add("https")
stopWords.add("type")
stopWords.add("quality")
stopWords.add("sign")
stopWords.add("size")
stopWords.add("wall")
stopWords.add("sun9")
stopWords.add("c_uniq_tag")
stopWords.add("userapi")
stopWords.add("impg")
stopWords.add("impf")

months = \
{
    "янв.": '01',
    "февр.": '02',
    "мар.": '03',
    "апр.": '04',
    "мая": '05',
    "июн.": '06',
    "июл.": '07',
    "авг.": '08',
    "сент.": '09',
    "окт.": '10',
    "нояб.": '11',
    "дек.": '12'
}


def makeNumpyDate(dateInp="23 янв. 2020", deleteDay=True):
    dateList = dateInp.split()
    if deleteDay:
        dateList[0] = "01"
    dateList[1] = months[dateList[1]]
    correctDate = dateList[2] + "-" + dateList[1] + "-" + f'{int(dateList[0]):02}'
    return np.datetime64(correctDate)


def parseTXT(filePath):
    fileInp = open(filePath, "r", encoding="UTF-8")
    listOfData = []

    for line in fileInp:
        if re.search(r"\(\d{1,2} .* \d{1,2}:\d{2}:\d{2} .{2}\):", line) is not None:
            line = line.strip()
            datePosInLine = re.search(r"\(\d{1,2} .* \d{1,2}:\d{2}:\d{2} .{2}\):", line).span()
            timePosInLine = re.search(r"\d{1,2}:\d{2}:\d{2} .{2}", line).span()

            name = line[:datePosInLine[0] - 1]
            dateOfMessage = line[datePosInLine[0] + 1:timePosInLine[0]]
            timeOfMessage = line[timePosInLine[0]:timePosInLine[1]]
            message = line[datePosInLine[1]:]

            dateOfMessage = makeNumpyDate(dateOfMessage)
            listOfData.append([name, dateOfMessage, timeOfMessage, message])

    data = pd.DataFrame(listOfData)
    data.columns = ["Name", "Date", "Time", "Message"]
    return data


def plotCountOfMessages(data):
    data2 = data.groupby(["Name", "Date"]).count().reset_index()
    data3 = pd.concat([data2[data2["Name"] == "Танюшка Баранова"],
                       data2[data2["Name"] == "Никита Луков"]])
    data3 = data3.drop(columns="Time")
    data3.columns = ["Name", "Date", "CountOfMessages"]
    # print(data3)
    sns.set_theme()
    ax = sns.lineplot(
        data=data3,
        x="Date", y="CountOfMessages", hue="Name",
    )
    ax.set(xticks=data3["Date"])
    plt.xticks(rotation=90)
    plt.show()


def plotWordCloudOfMessages(listOfMessagesOfPerson, nameOfOutputImage):
    stringOfText = ""
    for message in listOfMessagesOfPerson:
        stringOfText += message.strip()
    wordcloud = WordCloud(stopwords=stopWords, width=1920, height=1080).generate(stringOfText)

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(nameOfOutputImage, dpi=300)
    return 0


def getStatsAboutEmoji(listOfMessagesOfPerson, name="None"):
    usedEmoji = dict()
    for message in listOfMessagesOfPerson:
        for symbol in message:
            if int("1F600", 16) <= ord(symbol) <= int("1F64F", 16):  # is symbol is emoji
                if usedEmoji.get(symbol) is None:
                    usedEmoji[symbol] = 1
                else:
                    usedEmoji[symbol] += 1
    dataOfEmojis = pd.DataFrame(usedEmoji.items())
    dataOfEmojis.columns = ["Emoji", "Count"]
    dataOfEmojis.sort_values("Count", inplace=True, ascending=False)
    print(name + ":")
    print(dataOfEmojis.head())
    print(list(dataOfEmojis["Emoji"]))
    print(list(dataOfEmojis["Count"]))
    print()
    return 0


def getStatsAboutRussianSmile(listOfMessagesOfPerson, name="None"):  # )
    usedBrackets = dict([[")", 0], ["(", 0]])
    for message in listOfMessagesOfPerson:
        for symbol in message:
            if symbol == ")" or symbol == "(":
                usedBrackets[symbol] += 1
    dataOfBrackets = pd.DataFrame(usedBrackets.items())
    dataOfBrackets.columns = ["Bracket", "Count"]
    dataOfBrackets.sort_values("Count", inplace=True, ascending=False)
    print(name + ":")
    print(dataOfBrackets)
    print()


filePath = "../Datasets/private datasets/mesTatyana/FIX messages before 02.21.txt"
dataOfMessages = parseTXT(filePath)
# plotCountOfMessages(dataOfMessages)
# plotWordCloudOfMessages(dataOfMessages[dataOfMessages["Name"] == "Никита Луков"]["Message"],
#                         "wordCloud Nikita.jpeg")
# plotWordCloudOfMessages(dataOfMessages[dataOfMessages["Name"] == "Танюшка Баранова"]["Message"],
#                         "wordCloud Tatyana.jpeg")

# getStatsAboutEmoji(dataOfMessages[dataOfMessages["Name"] == "Никита Луков"]["Message"], name="Никита Луков")
# getStatsAboutEmoji(dataOfMessages[dataOfMessages["Name"] == "Танюшка Баранова"]["Message"], name="Танюшка Баранова")

getStatsAboutRussianSmile(dataOfMessages[dataOfMessages["Name"] == "Никита Луков"]["Message"], name="Никита Луков")
getStatsAboutRussianSmile(dataOfMessages[dataOfMessages["Name"] == "Танюшка Баранова"]["Message"], name="Танюшка Баранова")
