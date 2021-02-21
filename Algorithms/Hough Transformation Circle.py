import numpy as np
from time import time
from PIL import Image


def hougtTransformationCircles():
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

    load = Image.open("../Datasets/MyPicRecognition/Hough Circle3.png")
    documentSize = (load.size[0]//1, load.size[1]//1)
    load = load.resize(documentSize)
    imageArr = list(load.getdata())
    if type(imageArr[0]) is not int:
        for y in range(len(imageArr)):
            imageArr[y] = int(0.2989 * imageArr[y][0] + 0.5870 * imageArr[y][1] + 0.1140 * imageArr[y][2])
    imageArr = np.reshape(imageArr, documentSize[::-1])
    print(imageArr.shape)
    maxRadius = min(imageArr.shape[0], imageArr.shape[1])
    accumulateArray = np.zeros((imageArr.shape[0], imageArr.shape[1], maxRadius+1))
    pixelsWhichVoted = np.reshape([set() for _ in range(np.prod(accumulateArray.shape))], accumulateArray.shape)
    # a, b, r
    startCalc = time()

    checkBorderVector = np.vectorize(isPixelOnBorder)

    for x in range(2, imageArr.shape[0] - 2):
        print(x, end=";")
        if x % 50 == 0:
            print()
        for y in range(2, imageArr.shape[1] - 2):
            # пробегаем все точки картинки и смотрим, а эта точка- граница меж двух объектов?
            if isPixelOnBorder(x, y):
                allA, allB = np.meshgrid(np.arange(accumulateArray.shape[0]), np.arange(accumulateArray.shape[1]))
                allR = np.round(np.sqrt((x - allA)**2 + (y - allB)**2))
                allNextX = np.int32(np.round(np.sqrt(abs(allR**2 - (y + 1 - allB)**2)) + allA))
                flagsOfCorrectPX = allNextX + 1 < imageArr.shape[0]
                voted = checkBorderVector(allNextX, y + 1, flagsOfCorrectPX)
                votedParametersA = allA[voted]
                votedParametersB = allB[voted]
                votedParametersR = allR[voted]
                for a, b, r in zip(votedParametersA, votedParametersB, votedParametersR):
                    if r >= accumulateArray.shape[2]:
                        continue
                    accumulateArray[a][b][int(r)] += 1
                    pixelsWhichVoted[a][b][int(r)].add((y, x))
                    # somehow we need to save (y, x), not (x, y). Maybe i do not understand the coord system of PIL.
    print("\n Votes calculated")

    thresholdValue = 50  # (пороговое значение)
    countOfCircles = 0
    for a in range(accumulateArray.shape[0]):
        for b in range(accumulateArray.shape[1]):
            for r in range(accumulateArray.shape[2]):
                if accumulateArray[a][b][r] > thresholdValue:
                    countOfCircles += 1
                    xs = [pix[0] for pix in pixelsWhichVoted[a][b][r]]
                    ys = [pix[1] for pix in pixelsWhichVoted[a][b][r]]
                    drawPixels(xs, ys)

    maxA, maxB, maxR = np.unravel_index(np.argmax(accumulateArray), accumulateArray.shape)
    print("hougt calc in ", time() - startCalc, "secs")
    print("Found circles", countOfCircles)
    print("count of max vote:", np.max(accumulateArray))
    print("param of the largest circle", maxA, maxB, maxR)
    load.save("D:\\result.png")
    return maxA, maxB, maxR


if __name__ == "__main__":
    hougtTransformationCircles()
    print("end")
