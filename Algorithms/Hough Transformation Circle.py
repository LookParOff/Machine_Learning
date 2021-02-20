import numpy as np
from time import time
from PIL import Image


def hougtTransformationCircles(stepTheta=0.0175*2):
    def isPixelOnBorder(i_, j_, flagOfCorrectInput=True):
        nonlocal imageArr
        if not flagOfCorrectInput:
            return False
        px = imageArr[i_][j_]
        neiPx1 = imageArr[i_-1][j_]
        neiPx2 = imageArr[i_+1][j_]
        neiPx3 = imageArr[i_][j_-1]
        neiPx4 = imageArr[i_][j_+1]

        diffPix = 50  # мера, на сколько должны отличаться цвета, чтобы это была граница объекта
        if abs(px - neiPx1) > diffPix or \
            abs(px - neiPx2) > diffPix or \
            abs(px - neiPx3) > diffPix or \
            abs(px - neiPx4) > diffPix:
            return True
        return False

    def drawPixels(kitOfX, kitOfY):
        for xi, yi in zip(kitOfX, kitOfY):
            if 0 < yi < imageArr.shape[0] and 0 < xi < imageArr.shape[1]:
                load.putpixel((xi, yi), (255, 0, 0))

    load = Image.open("../Datasets/MyPicRecognition/Hough Circle1.png")
    documentSize = (load.size[0]//1, load.size[1]//1)
    load = load.resize(documentSize)
    imageArr = list(load.getdata())
    if type(imageArr[0]) is not int:
        for y in range(len(imageArr)):
            imageArr[y] = int(0.2989 * imageArr[y][0] + 0.5870 * imageArr[y][1] + 0.1140 * imageArr[y][2])
    imageArr = np.reshape(imageArr, documentSize)
    print(imageArr.shape)
    maxRadius = min(imageArr.shape[0], imageArr.shape[1])
    accumulateArray = np.zeros((imageArr.shape[0], imageArr.shape[1], maxRadius*2))
    pixelsWhichVoted = np.reshape([set() for _ in range(np.prod(accumulateArray.shape))], accumulateArray.shape)
    # a, b, r
    startCalc = time()

    checkBorderVector = np.vectorize(isPixelOnBorder)
    for y in range(2, imageArr.shape[0] - 2):
        print(y)
        for x in range(2, imageArr.shape[1] - 2):
            # пробегаем все точки картинки и смотрим, а эта точка- граница меж двух объектов?
            if isPixelOnBorder(y, x):
                allA, allB = np.meshgrid(np.arange(accumulateArray.shape[0]), np.arange(accumulateArray.shape[1]))
                allR = np.round(np.sqrt((x - allA)**2 + (y - allB)**2))
                allNextX = np.int32(np.round(np.sqrt(abs(allR**2 - (y + 1 - allB)**2)) + allA))
                allPrevX = np.int32(np.round(np.sqrt(abs(allR ** 2 - (y - 1 - allB) ** 2)) + allA))
                flagsOfCorrectPX1 = allNextX + 1 < imageArr.shape[0]
                flagsOfCorrectPX2 = allPrevX + 1 < imageArr.shape[0]
                voted = checkBorderVector([y + 1] * len(allNextX), allNextX, flagsOfCorrectPX1)
                votedParametersA = allA[voted]
                votedParametersB = allB[voted]
                votedParametersR = allR[voted]
                for a, b, r in zip(votedParametersA, votedParametersB, votedParametersR):
                    accumulateArray[a][b][int(r)] += 1
                    pixelsWhichVoted[a][b][int(r)].add((x, y))
    print("Votes calculated")
    maxVote = np.max(accumulateArray)
    print("maxVote", maxVote)
    thresholdValue = 50  # (пороговое значение)

    maxA, maxB, maxR = np.unravel_index(np.argmax(accumulateArray), accumulateArray.shape)

    # xs = np.arange(imageArr.shape[0])
    # ys = [int(round(np.sqrt(abs(maxR ** 2 - (xi + 1 - maxA) ** 2)) + maxB)) for xi in xs]
    # drawPixels(xs, ys)
    # xs = np.arange(imageArr.shape[0])
    # ys = [int(round(-np.sqrt(abs(maxR ** 2 - (xi + 1 - maxA) ** 2)) + maxB)) for xi in xs]
    xs = [pix[0] for pix in pixelsWhichVoted[maxA][maxB][maxR]]
    ys = [pix[1] for pix in pixelsWhichVoted[maxA][maxB][maxR]]
    drawPixels(xs, ys)

    print("hougt calc in ", time() - startCalc, "secs")
    print(maxA, maxB, maxR)

    load.save("D:\\result.png")
    return maxA, maxB, maxR


if __name__ == "__main__":
    hougtTransformationCircles()
    print("end")
