import numpy as np
import mnist
import time
from myNeuralNetwork import Network
import random


net = Network([784, 10, 10])
trainImages = mnist.train_images()
testImages = mnist.test_images()

answerOfTrainImages = mnist.train_labels()
answerOfTestImages = mnist.test_labels()
resultOfTrainImages = []

for rightLabel in answerOfTrainImages:
    resultOfNet = np.array([[int(i == rightLabel)] for i in range(10)])
    resultOfTrainImages.append(resultOfNet)

resultOfTrainImages = np.array(resultOfTrainImages)

trainImages = np.reshape(trainImages, (60000, 28 * 28, 1)) / 255
testImages = np.reshape(testImages, (10000, 28 * 28, 1)) / 255


s = time.time()
net.train(trainImages, resultOfTrainImages, testImages, answerOfTestImages, 10, 0.5)
print("time of train", abs(s - time.time()))
