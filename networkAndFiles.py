import numpy as np


def saveToFile(network, fileName):
    file = open("NeuralNets\\" + fileName, "w")
    for layer in network.weights:
        for wFromOneNeuron in layer:
            for w in wFromOneNeuron:
                file.write(str(w) + " ")
            file.write("\n")
        file.write('\n')
    file.write('biases\n')
    for layer in network.biases:
        for biases in layer:
            for bias in biases:
                file.write(str(bias) + " ")
        file.write("\n")
    file.close()


def loadFromFile(network, fileName):
    file = open("NeuralNets\\" + fileName, "r")
    weights = [[]]
    biases = []
    isW = True
    indexOfLayer = 0
    for line in file:
        data = line.strip("\n").split()
        if isW and len(data) == 0:
            weights[-1] = np.array(weights[-1])
            weights.append([])
            continue
        if line == "biases\n":
            isW = False
            weights.pop(-1)
            continue
        if not isW and len(data) == 0:
            continue

        for i in range(len(data)):
            data[i] = float(data[i])
        data = np.array(data)
        if isW:
            weights[-1].append(data)
        else:
            biases.append(np.reshape(data, (len(data), 1)))
    file.close()
    network.weights = weights
    network.biases = biases
