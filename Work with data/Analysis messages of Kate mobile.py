from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
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


def makeFixedFile():
    # make one message in one row
    fileInp = open("..//Datasets/private datasets/mesTatyana/messages before 02.21.txt", "r", encoding="UTF-8")
    fileFixed = open("..//Datasets/private datasets/mesTatyana/FIX messages before 02.21.txt", "w", encoding="UTF-8")
    attachments = False
    for line in fileInp:
        if len(line.strip()) == 0:
            attachments = False
            continue
        if re.search("[а-яА-ЯёЁ]{3,} [а-яА-ЯёЁ]{3,} \(\d{1,2} .* \d{1,2}:\d{2}:\d{2} .{2}\):", line) is not None:
            fileFixed.write("\n" + line.strip() + " ")
        else:
            fileFixed.write(line.strip() + " ")
    print("end")


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
            if int("1F600", 16) <= ord(symbol) <= int("1F64F", 16) or ord(symbol) == 127773 or ord(symbol) == 127770:
                # is symbol is emoji or emoji of moon
                if usedEmoji.get(symbol) is None:
                    usedEmoji[symbol] = 1
                else:
                    usedEmoji[symbol] += 1
    dataOfEmojis = pd.DataFrame(usedEmoji.items())
    dataOfEmojis.columns = ["Emoji", "Count"]
    dataOfEmojis.sort_values("Count", inplace=True, ascending=False)
    print(dataOfEmojis)
    fig = px.pie(dataOfEmojis, values="Count", names='Emoji')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()
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


def plotMessagesOfHours(listOfMessagesOfPerson):
    allTimeOfMessages = listOfMessagesOfPerson["Time"]
    counts = [0 for _ in range(24 * 60)]
    ticksForGraph = {'{:02}'.format(i // 60) + ":" + '{:02}'.format(i % 60): 0 for i in range(24 * 60)}
    for time in allTimeOfMessages:
        # time 11:38:14 ПП
        timeList = time.split(":")
        timeList = timeList[:-1] + timeList[-1].split(" ")
        added12Hours = 0
        if timeList[-1] == "ПП":  # after midday
            added12Hours = 12
        timeList = ((int(timeList[0]) % 12) + added12Hours, int(timeList[1]), int(timeList[2]))
        counts[timeList[0] * 60 + timeList[1]] += 1
        ticksForGraph[f"{timeList[0]:02}:{timeList[1]:02}"] += 1
    ax = sns.lineplot(data=ticksForGraph)
    ax.set(xticks=list(ticksForGraph.keys())[::130],
           xlabel="время", ylabel="кол-во сообщений",
           title="Распределение сообщений по времени суток")
    # plt.xticks(rotation=90)
    plt.show()
    return 0


filePath = "../Datasets/private datasets/mesTatyana/FIX messages before 02.21.txt"
dataOfMessages = parseTXT(filePath)
# print(dataOfMessages.sample(5))
# plotCountOfMessages(dataOfMessages)
# plotWordCloudOfMessages(dataOfMessages[dataOfMessages["Name"] == "Никита Луков"]["Message"],
#                         "wordCloud Nikita.jpeg")
# plotWordCloudOfMessages(dataOfMessages[dataOfMessages["Name"] == "Танюшка Баранова"]["Message"],
#                         "wordCloud Tatyana.jpeg")

# getStatsAboutEmoji(dataOfMessages[dataOfMessages["Name"] == "Никита Луков"]["Message"], name="Никита Луков")
# getStatsAboutEmoji(dataOfMessages[dataOfMessages["Name"] == "Танюшка Баранова"]["Message"], name="Танюшка Баранова")

# getStatsAboutRussianSmile(dataOfMessages[dataOfMessages["Name"] == "Никита Луков"]["Message"], name="Никита Луков")
# getStatsAboutRussianSmile(dataOfMessages[dataOfMessages["Name"] == "Танюшка Баранова"]["Message"],
#                                                                                           name="Танюшка Баранова")

# plotMessagesOfHours(dataOfMessages[dataOfMessages["Date"] == np.datetime64("2020-02-01")])
# plotMessagesOfHours(dataOfMessages)
