import matplotlib.pyplot as plt
import numpy as np

# https://habr.com/ru/post/468295/


def generateData():
    x = np.arange(0, 1, 0.01)
    y = []
    for xi in x:
        y.append(np.random.random()*10 + 10)
    y = np.array(y)
    return x, y


def distance(x0, y0, k, b):
    return abs(-k*x0 + y0 - b) / (k**2 + 1)**0.5


x, y = generateData()
plt.scatter(x, y, 5)

# y = kx + b
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
plt.plot([x[0], x[-1]], [x[0]*k + b, x[-1]*k + b], color="black", linestyle="-", linewidth=2)
plt.gca().set(xlim=(x[0]-5, x[-1]+5), ylim=(min(y)-5, max(y)+5))
plt.show()
