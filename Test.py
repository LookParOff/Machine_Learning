import numpy as np
import time
import random
from myNeuralNetwork import Network
from networkAndFiles import saveToFile, loadFromFile


def fromBinToDec(a):
    aList = []
    for digit in str(a):
        aList.append(int(digit))
    aList = aList[::-1]
    degree = 0
    res = 0
    for digit in aList:
        res += digit * 2**degree
        degree += 1
    listRes = []
    for digit in str(res):
        listRes.append(int(digit))
    return listRes


def fromDecToBinN(a, N):
    listRes = []
    for digit in bin(a)[2:]:
        listRes.append([int(digit)])
    while len(listRes) < N:
        listRes.insert(0, [0])
    if len(listRes) != N:
        print("Err", len(listRes))
    return listRes


class TestNetwork(Network):
    def checkCorrectOfAnswer(self, test, answers):
        correct = 0
        wrong = 0
        for num, answer in zip(test, answers):
            resOfNet = self.compute(num)
            delta = abs(resOfNet - answer)
            if max(delta) > 0.3:
                wrong += 1
            else:
                correct += 1
        print("Correct:", correct)
        print("Wrong:", wrong)
        print("Correct%:", round(correct/(correct + wrong) * 100, 2), "%\n")


net = TestNetwork([16, 1024, 16])
nums = [(random.randint(0, 2**8-1), random.randint(0, 2**8-1)) for _ in range(2**16)]

data = np.array([fromDecToBinN(el[0], 8) + fromDecToBinN(el[1], 8) for el in nums])
dataRes = np.array([fromDecToBinN(el[0] * el[1], 16) for el in nums])

# indices = [i for i in range(2**16-1)]
# random.shuffle(indices)
# data = data[indices]
# dataRes = dataRes[indices]

amountOfData = 50000
train = data[:amountOfData]
trainRes = dataRes[:amountOfData]
test = data[amountOfData:]
testRes = dataRes[amountOfData:]

# loadFromFile(net, "multOfbin_8bit.txt")
# print(fromDecToBinN(2**8-1, 8))
# print(net.compute(fromDecToBinN(2**8-1, 8) + fromDecToBinN(2**8-1, 8)))
net.train(train, trainRes, test, testRes, 30, 1)
saveToFile(net, "shittyWordGen.txt")
