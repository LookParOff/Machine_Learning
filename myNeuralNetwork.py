import numpy as np
import random
import time
# np.seterr("raise")
# all warnings will be exceptions. I thing it's useful
# if we will fix warnings of numpy we will bypass strong errors in future


# This was wrote in 22 days ;)
class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(countOfNeuron, countOfEdges)
                        for countOfEdges, countOfNeuron in zip(sizes[:-1], sizes[1:])]

    def checkCorrectOfAnswer(self, test, answers):
        correct = 0
        wrong = 0
        for image, answer in zip(test, answers):
            res = self.evaluate(image)[1][-1]
            if max(res) == res[answer] > 0.5 and sum(res) < 5.5:
                # second stmt means, what all other outputs lower than 0.5. 0.5*9 + 1- edge accepted
                correct += 1
            else:
                wrong += 1
        print("Correct:", correct)
        print("Wrong:", wrong)
        print("\n")

    def compute(self, inputActivation):
        activation = inputActivation.copy()
        for w, b in zip(self.weights, self.biases):
            z = np.reshape(np.dot(w, activation), (len(b), 1)) + b
            activation = sigmoid(z)
        return np.round(activation, 3)

    def evaluate(self, inputActivation):
        # inputActivation- input vector
        activation = inputActivation.copy()
        zs = []
        activations = [activation]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b  # deleted reshape
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return zs, activations

    def evaluateOnMiniBatch(self, miniBatch, results, learningRate):
        nablaOfWeights = []
        nablaOfBias = []
        arrForCalcAverageOfNablaW = []
        arrForCalcAverageOfNablaB = []
        for inputActivation, result in zip(miniBatch, results):
            zs, activations = self.evaluate(inputActivation)
            delta = [self.derivativeOfCostFunc(result, activations[-1])]
            # delta = [(activations[-1] - result) * sigmoid_prime(zs[-1])]
            nablaOfBias = [delta[-1]]
            nablaOfWeights = [np.dot(delta[-1], np.transpose(activations[-2]))]
            for l in range(2, self.num_layers):
                # подумать над транспонированием np.transpose
                delta.append(np.matmul(np.transpose(self.weights[-l + 1]), delta[-1]) * sigmoid_prime(zs[-l]))
                nablaOfWeights.append(delta[-1] * np.transpose(activations[-l - 1]))
                nablaOfBias.append(delta[-1])
            nablaOfWeights = nablaOfWeights[::-1]
            nablaOfBias = nablaOfBias[::-1]
            arrForCalcAverageOfNablaW.append(nablaOfWeights)
            arrForCalcAverageOfNablaB.append(nablaOfBias)

        resNablaW = [np.zeros(w.shape) for w in self.weights]
        resNablaB = [np.zeros(b.shape) for b in self.biases]
        # calc the average nabla
        for nablaW in arrForCalcAverageOfNablaW:
            for index, layer in enumerate(nablaW):
                resNablaW[index] += layer
        for nablaB in arrForCalcAverageOfNablaB:
            for index, layer in enumerate(nablaB):
                resNablaB[index] += layer

        for index, _ in enumerate(resNablaW):
            resNablaW[index] /= len(arrForCalcAverageOfNablaW)
        for index, _ in enumerate(resNablaB):
            resNablaB[index] /= len(arrForCalcAverageOfNablaB)

        # correct the weights
        for index, nablaW in enumerate(resNablaW):
            self.weights[index] -= nablaW * learningRate
        for index, nablaB in enumerate(resNablaB):
            self.biases[index] -= nablaB * learningRate
        return 0

    def train(self, train, trainAnswers, test, testAnswers, epochs, learningRate):
        # print("Random net generate a random shit:")
        # self.checkCorrectOfAnswer(test, testAnswers)
        startTimeTrain = time.time()
        for epoch in range(epochs):
            # shuffle the data
            # indices = np.arange(train.shape[0])
            # np.random.shuffle(indices)
            # train = train[indices]
            # trainAnswers = trainAnswers[indices]

            lenOfBatch = 100
            miniBatches = []
            batchesOfCorrectionData = []
            timeEpoch = time.time()
            for k in range(0, len(train), lenOfBatch):
                miniBatches.append(train[k: k + lenOfBatch])
                batchesOfCorrectionData.append(trainAnswers[k: k + lenOfBatch])

            for miniBatch, correct in zip(miniBatches, batchesOfCorrectionData):
                self.evaluateOnMiniBatch(miniBatch, correct, learningRate)

            print("Train iteration is", epoch, ":")
            self.checkCorrectOfAnswer(test, testAnswers)
            t = round(time.time() - timeEpoch)
            print("This epoch was calc in", t, "sec\n")
        t = round(time.time() - startTimeTrain)
        print("This network was trained in", t, "sec")

    def derivativeOfCostFunc(self, y, a):
        # Actually this is not just derivative of Cost. We mult in on da/dz, i.e. on sigmoid_prime
        return a - y  # cross validation cost


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    # Derivative of the sigmoid function.
    return sigmoid(z)*(1-sigmoid(z))
