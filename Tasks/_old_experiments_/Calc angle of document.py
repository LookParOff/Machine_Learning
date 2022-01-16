from Algorithms.myNeuralNetwork import Network
import numpy as np
from PIL import Image
from Tasks.networkAndFiles import saveToFile, loadFromFile

PATH_IMAGES = "../Datasets/scan_doc_rotation/images/"
PATH_LABELS = "../Datasets/scan_doc_rotation/labels/"
namesOfTrainImg = open("../Datasets/scan_doc_rotation/train_list.json", "r")
arrOfImages = []
arrOfLabels = []

N = 400
for name in namesOfTrainImg:
    if len(name) > 4:
        name = name.split()[0][1:-1]
        if name[-1] == '"':
            name = name[:-1]
        pathOnCurrImg = PATH_IMAGES + name
        img = Image.open(pathOnCurrImg)
        arrOfImages.append(img)
        fileOfLabel = open(PATH_LABELS + name[:-3] + "txt")
        label = round(float(fileOfLabel.read().splitlines()[0]), 0)
        arrOfLabels.append(int(label)+5)
        fileOfLabel.close()
    if len(arrOfLabels) == N:
        break

namesOfTrainImg.close()
countOfClasses = 11
for i in range(len(arrOfImages)):
    img = arrOfImages[i].getdata()
    arrOfImages[i] = np.reshape(img, (len(img), 1))
    answer = arrOfLabels[i]
    arrOfLabels[i] = np.array([[0]]*countOfClasses)
    arrOfLabels[i][answer] = 1


train = np.reshape(arrOfImages, (len(arrOfImages), len(arrOfImages[0]), 1)) / 255
trainLabels = np.reshape(arrOfLabels, (len(arrOfLabels), countOfClasses, 1))
arrOfImages.clear()
arrOfLabels.clear()
print(train[0])
print(train.shape)
print(trainLabels[0])
print(trainLabels.shape)


class NetworkForDocs(Network):
    def checkCorrectOfAnswer(self, test, answers):
        correct = 0
        wrong = 0
        loss = 0
        for image, answer in zip(test, answers):
            res = self.evaluate(image)[1][-1]
            loss += ((res - answer)**2).sum()
            ansIndex = answer.argmax()
            if max(res) == res[ansIndex] > 0.5 and sum(res) < 5.5:
                # second stmt means, what all other outputs lower than 0.5. 0.5*9 + 1- edge accepted
                correct += 1
            else:
                wrong += 1
        loss /= len(test)
        print("Correct:", correct)
        print("Wrong:", wrong)
        print("acc", correct / (correct + wrong))
        print("loss", loss)
        print()


net = NetworkForDocs((len(train[0]), 512, countOfClasses))
# loadFromFile(net, "CalculationOfAngleOfDocument.txt")
amountOfTrain = N - N//3
net.train(train[:amountOfTrain], trainLabels[:amountOfTrain],
          train[amountOfTrain:], trainLabels[amountOfTrain:],
          epochs=10, learningRate=0.5)
print("start change type of nums")
for i in range(len(net.weights)):
    net.weights[i] = np.single(net.weights[i])
    net.biases[i] = np.single(net.biases[i])
print("start saving")
saveToFile(net, "CalculationOfAngleOfDocument.txt")
