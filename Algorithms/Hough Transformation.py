import numpy as np
from time import time
from PIL import Image, ImageTk
from collections.abc import Iterable


def hougtTransformation(stepTheta=0.0175*2):
    def isPixelOnBorder(i_, j_):
        nonlocal imageArr
        px = imageArr[i_][j_]
        neiPx1 = imageArr[i_-1][j_]
        neiPx2 = imageArr[i_+1][j_]
        neiPx3 = imageArr[i_][j_-1]
        neiPx4 = imageArr[i_][j_+1]
        if isinstance(px, Iterable):
            px = px[0]
            neiPx1 = neiPx1[0]
            neiPx2 = neiPx2[0]
            neiPx3 = neiPx3[0]
            neiPx4 = neiPx4[0]
        diffPix = 50  # мера, на сколько должны отличаться цвета, чтобы это была граница объекта
        if abs(px - neiPx1) > diffPix or \
            abs(px - neiPx2) > diffPix or \
            abs(px - neiPx3) > diffPix or \
            abs(px - neiPx4) > diffPix:
            return True
        return False

    def drawLine(kitOfX, kitOfY):
        for xi, yi in zip(kitOfX, kitOfY):
            if 0 < yi < imageArr.shape[0] and 0 < xi < imageArr.shape[1]:
                load.putpixel((xi, yi), (255, 0, 0))

    load = Image.open("C:\\Hough SQUARE.png")
    documentSize = (load.size[0]//1, load.size[1]//1)
    load = load.resize(documentSize)
    print(load.size)
    # some ways to upgrade the accuracy of finding lines:
    # OperationSobel(True)
    # OperationKirsha(True)

    imageArr = list(load.getdata())
    if type(imageArr[0]) is not int:
        for i in range(len(imageArr)):
            imageArr[i] = int(0.2989 * imageArr[i][0] + 0.5870 * imageArr[i][1] + 0.1140 * imageArr[i][2])

    imageArr = np.reshape(imageArr, load.size[::-1])
    print(imageArr.shape)
    thetas = np.arange(0.01, 3.14, stepTheta)
    countOfRo = int((imageArr.shape[0]**2 + imageArr.shape[1] ** 2)**0.5)
    accumulateArray = np.zeros((countOfRo, len(thetas)))
    # remember all pixel (x, y), which voted for line
    pixelsWhichVoted = np.reshape([set() for _ in range(countOfRo*len(thetas))], accumulateArray.shape)

    # x*cos(theta) + y*sin(theta) = ro
    # y = (ro - x*cos(theta))/ sin(theta)
    # y = ro / sin(theta) - x*ctg(theta)
    # x = ro / cos(theta) - y * tg(theta)
    startCalc = time()
    for i in range(2, imageArr.shape[0] - 2):
        print(i)
        for j in range(2, imageArr.shape[1] - 2):
            # пробегаем все точки картинки и смотрим, а эта точка- граница меж двух объектов?
            if isPixelOnBorder(i, j):
                # start search every line, what could be
                for theta in thetas:
                    # j = column = x
                    ro = j*np.cos(theta) + i*np.sin(theta)
                    # now see is line theta and ro match with line on image
                    # will see next column j+1 the row (ro / sin(theta) - x*ctg(theta))

                    y = int(round(ro / np.sin(theta) - (j+1)*(1/np.tan(theta)), 0))
                    # print(y, end="|")
                    if not 0 < y < imageArr.shape[0] - 1:
                        # y coord in next column is not in image, that's mean line is about vertical
                        x = int(round(ro / np.cos(theta) - (i + 1) * np.tan(theta), 0))  # coord of next pixel on the line
                        if isPixelOnBorder(i + 1, x):
                            ro = int(round(ro, 0))
                            indexTheta = np.where(thetas == theta)[0][0]
                            accumulateArray[ro][indexTheta] += 1
                            pixelsWhichVoted[ro][indexTheta].add((x, i))
                    else:
                        if isPixelOnBorder(y, j + 1):
                            ro = int(round(ro, 0))
                            indexTheta = np.where(thetas == theta)[0][0]
                            accumulateArray[ro][indexTheta] += 1
                            pixelsWhichVoted[ro][indexTheta].add((j, y))

    print("Votes calculated")
    maxVote = np.max(accumulateArray)
    print("maxVote", maxVote)

    thresholdValue = 50  # (пороговое значение)
    parametersOfPickedLines = []
    drawLineOfNot = []
    for i in range(accumulateArray.shape[0]):
        for j in range(accumulateArray.shape[1]):
            if accumulateArray[i][j] > thresholdValue:
                ro = i
                indexTheta = j
                theta = thetas[j]
                parametersOfPickedLines.append((ro, indexTheta))
                drawLineOfNot.append(True)

    for i in range(len(parametersOfPickedLines)):
        for j in range(len(parametersOfPickedLines)):
            if i != j and drawLineOfNot[i] and drawLineOfNot[j]:
                if abs(parametersOfPickedLines[i][0] - parametersOfPickedLines[j][0]) < 15 \
                        and abs(parametersOfPickedLines[i][1] - parametersOfPickedLines[j][1]) < 15:
                    # check, all combinations of lines and do not draw lines with same parameters
                    drawLineOfNot[i] = False

    countOfFoundedLines = 0
    thetaVetric = 0
    for indexOfLine in range(len(parametersOfPickedLines)):
        if drawLineOfNot[indexOfLine]:
            # draw a line
            ro, indexTheta = parametersOfPickedLines[indexOfLine]
            theta = thetas[indexTheta]
            if theta < 0.106:
                thetaVetric = theta
            # allPixels = pixelsWhichVoted[ro][indexTheta]
            # xs = [pix[0] for pix in allPixels]
            # ys = [pix[1] for pix in allPixels]

            xs = [i for i in range(imageArr.shape[1])]
            ys = [int(round(ro / np.sin(theta) - x*(1/np.tan(theta)), 0)) for x in xs]
            drawLine(xs, ys)
            ys = [i for i in range(imageArr.shape[0])]
            xs = [int(round(ro / np.cos(theta) - y * np.tan(theta), 0)) for y in ys]
            drawLine(xs, ys)

            countOfFoundedLines += 1
    print("hougt calc in ", time() - startCalc, "secs")
    print("count of founded fines", countOfFoundedLines)

    HafSpace = Image.fromarray(accumulateArray)
    load.save("D:\\result.png")
    ro, indexTheta = np.unravel_index(np.argmax(accumulateArray), accumulateArray.shape)
    theta = thetas[indexTheta]
    return thetaVetric, theta
    # return ro, theta


hougtTransformation()