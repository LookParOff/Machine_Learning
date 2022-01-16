from Algorithms.myNeuralNetwork import Network
from Programs.networkAndFiles import loadFromFile


# will take 10 symbols on one word
# 5 neuron on coding one symbol
# 50 neurons on input


def isWordRussian(word):
    for symb in word:
        if symb not in RUSSIAN_ALFABET:
            return False
    return True


def fromDecToBinN(a, N):
    listRes = []
    for digit in bin(a)[2:]:
        listRes.append([int(digit)])
    while len(listRes) < N:
        listRes.insert(0, [0])
    return listRes


def fromWordToCod(word):
    cod = []
    for symb in word:
        cod += codedSymb[symb]
    while len(cod) < 50:
        cod += [[0]]
    return cod


def fromCodToWord(codedWord):
    res = ""
    lenOfOneCodedSymb = 5
    for i in range(0, len(codedWord) - lenOfOneCodedSymb, lenOfOneCodedSymb):
        codedSymb = codedWord[i:i + lenOfOneCodedSymb]
        codedSymbSTR = ""
        for digit in codedSymb:
            codedSymbSTR += str(digit)
        symb = decodedSymb[codedSymbSTR]
        res += symb
    return res


def makeData(fileName):
    file = open("data/" + fileName, "r", encoding="UTF-8")
    data = []
    dataRes = []

    ii = 0
    for line in file:
        line = line.split()
        for index in range(len(line) - 1):
            word = line[index]
            nextWord = line[index+1]
            currCod = fromWordToCod(word)
            nextCod = fromWordToCod(nextWord)
            data.append(currCod)
            dataRes.append(nextCod)
            ii += 1
            if ii > 10:
                break
    file.close()
    return data, dataRes


class LangNetwork(Network):
    def computeAndRound(self, inputActivation):
        res = list(self.compute(inputActivation))
        for i in range(len(inputActivation)):
            res[i] = int(round(res[i][0]))
        return res

    def checkCorrectOfAnswer(self, test=0, answers=0):
        print(fromCodToWord(self.computeAndRound(fromWordToCod("сударь"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("но"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("позвольте"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("адъютант"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("да"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("это"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("лев"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("граф"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("нет"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("да"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("вы"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("багратион"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("кутузов"))))
        print(fromCodToWord(self.computeAndRound(fromWordToCod("наташа"))))


letters = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".lower()
RUSSIAN_ALFABET = set(list(letters))
codedSymb = dict()
decodedSymb = dict()
for index in range(32):
    codedSymb[letters[index]] = fromDecToBinN(index, 5)
codedSymb["ё"] = codedSymb["е"]

for key, value in codedSymb.items():
    cod = ""
    for digit in value:
        cod += str(digit[0])
    decodedSymb[cod] = key


train, trainRes = makeData("Improved War and peace.txt")
net = LangNetwork([50, 100, 100, 50])
loadFromFile(net, "shittyWordGen.txt")
net.checkCorrectOfAnswer()
print("START")
# net.train(train, trainRes, [], [], epochs=3, learningRate=3)
# saveToFile(net, "shittyWordGen.txt")

# TODO cod foocking space; maybe need to filter data and keep only 4-6 symb words
# todo есть идея взять много произведений и сделать файл, содержащий только 5-ти буквенные словосочетания из них
