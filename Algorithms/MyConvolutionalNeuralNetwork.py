import numpy as np


class ConvolutionLayer:
    def __init__(self, countOfFilters, countOfChannels, shapeOfFilter):
        # r- count of conv filters, k- count of channels, shapeF- shape of filter
        #                           (r   k shapeF)
        # [(8, (3, 5, 5)), (1, (8, 3, 3))]
        self.shapesOfConvFilter = (countOfFilters, (countOfChannels, shapeOfFilter[0], shapeOfFilter[1]))
        self.convFilters = [np.random.random_sample(self.shapesOfConvFilter[1]) for _ in range(self.shapesOfConvFilter[0])]
        self.biases = np.random.random(self.shapesOfConvFilter[0])

    def evaluateLayer(self, inputActivation):
        activation = inputActivation.copy()
        z = []
        for convFilter, bias in zip(self.convFilters, self.biases):
            z.append(convolveTensors(activation, convFilter) + bias)
        activation = functionActivationReLu(np.array(z))
        return activation

    def backPropagation(self, activationOfLayer, nablaOfActivation, outputOfPreviousLayer, learningRate=0.1):
        return 0

    def nablaOfActivationOfLayer(self):
        return 0

    def nablaOfOutputOfLayer(self):
        return 0

    def nablaOfWeightsOfLayer(self):
        return 0

    def nablaOfBiasOfLayer(self):
        return 0


def functionActivationReLu(z):
    mask = z > 0
    mask = np.int32(mask)
    z = z * mask
    return z


def convolveMatrices(inpMat: np.ndarray, convFilter: np.ndarray):
    resultMatrix = np.zeros((inpMat.shape[0] - convFilter.shape[0] + 1, inpMat.shape[1] - convFilter.shape[1] + 1))
    inpShape = inpMat.shape
    filterShape = convFilter.shape
    for iRow in range(inpShape[0] - filterShape[0] + 1):
        for iCol in range(inpShape[1] - filterShape[1] + 1):
            currA = inpMat[iRow:iRow + filterShape[0], iCol:iCol + filterShape[1]]
            resultMatrix[iRow][iCol] = np.sum(currA * convFilter)
    return resultMatrix


def convolveTensors(a: np.ndarray, b: np.ndarray):
    # a and b maybe be a tensors, so we convolve an each matrices in it
    if len(a.shape) != len(b.shape):
        raise Exception("someone have channels, someone- not. Is this ok?")
    if len(a.shape) == len(b.shape) == 3 and a.shape[0] != b.shape[0]:
        raise Exception("count of channels aren't equal. Is this ok?")
    if len(a.shape) == 2:
        return convolveMatrices(a, b)
    else:
        countOfChannels = a.shape[0]
        resultOfConvolve = convolveMatrices(a[0], b[0])
        for i in range(1, countOfChannels):
            resultOfConvolve += convolveMatrices(a[i], b[i])
        return resultOfConvolve
