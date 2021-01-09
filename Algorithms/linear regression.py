import matplotlib.pyplot as plt
import numpy as np
import time
# https://habr.com/ru/post/468295/


def generateData(startX, endX, step, spread=10, bias=10, hasAngle=True):
    x = np.arange(startX, endX, step)
    y = []
    for xi in x:
        if hasAngle:
            y.append(xi*np.random.random()*spread + bias)
        else:
            y.append(np.random.random() * spread + bias)
    y = np.array(y)
    return x, y


def distance(x0, y0, k, b):
    return abs(-k*x0 + y0 - b) / (k**2 + 1)**0.5


def linReg1(x, y):
    # cost func is func of distance
    k = 0
    b = 0
    epochs = 10000
    N = len(x)

    learningRate = 10
    colorInd = 0
    colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    previousError = 2**64
    for epoch in range(epochs):
        nablaK = 0
        nablaB = 0
        error = 0
        for xi, yi in zip(x, y):
            error += distance(xi, yi, k, b) ** 2
            nablaK += ((-k*xi + yi - b) / (abs(-k*xi + yi - b)) * -xi * np.sqrt(k**2 + 1) - abs(k * xi - yi + b) * k / np.sqrt(k**2 + 1)) / (k**2 + 1)
            nablaB += -1 / np.sqrt(k**2 + 1) * (-k*xi + yi - b) / (abs(-k*xi + yi - b))

        error /= N
        if previousError < error and abs(error - previousError) > 1:
            learningRate /= 10
            print("CHANGE lr, epoch =", epoch, "because", previousError, error, abs(error - previousError), "\n")
        if abs(previousError - error) < 0.0000001 and error < 1:
            print(epoch, "stop because it's stoped")
            break
        k -= (nablaK / N) * learningRate
        b -= (nablaB / N) * learningRate
        previousError = error
    print("error", error)
    return k, b


def linReg2(x, y):
    # cost func is diffrence between yi and line
    N = len(x)
    k = 100
    b = 0
    epochs = 20000
    learningRate = 10
    colorInd = 0
    colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    previousError = float("inf")
    for epoch in range(epochs):
        nablaK = 0
        nablaB = 0
        error = 0
        for xi, yi in zip(x, y):
            error += (yi - (xi*k + b)) ** 2
            nablaK += (yi - (xi*k + b)) * xi
            nablaB += (yi - (xi*k + b))

        error /= N
        if previousError < error and abs(error - previousError) > 1:
            learningRate /= 10
            print("CHANGE lr, epoch =", epoch, "because", previousError, error, abs(error - previousError), "\n")
        if abs(previousError - error) < 0.0000001 and error < 1:
            print(epoch, "stop because it's stoped")
            break
        k -= (-2 * nablaK / N) * learningRate
        b -= (-2 * nablaB / N) * learningRate
        previousError = error
        # plt.scatter(x, y, 5)
        # plt.plot([x[0], x[-1]], [x[0] * k + b, x[-1] * k + b], color="black", linestyle="-", linewidth=2)
        # plt.gca().set(xlim=(x[0] - 5, x[-1] + 5), ylim=(min(y) - 5, max(y) + 5))
        # plt.show()
        # sleep(1)
    print("error", error)
    return k, b


x, y = generateData(1, 2, 0.005, spread=100, hasAngle=True)
# y = kx + b
start = time.time()
k, b = linReg2(x, y)
print("time", time.time() - start)
plt.scatter(x, y, 5)
plt.plot([x[0], x[-1]], [x[0]*k + b, x[-1]*k + b], color="black", linestyle="-", linewidth=2)
plt.gca().set(xlim=(x[0]-5, x[-1]+5), ylim=(min(y)-5, max(y)+5))
plt.show()
