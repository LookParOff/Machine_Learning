import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly
import seaborn as sns
import matplotlib.pyplot as plt

pio.renderers.default = "browser"

# https://habr.com/ru/post/468295/


def generate_data(start_x, end_x, step=0.1, spread=10, bias=0):
    x = np.arange(start_x, end_x, step)
    y = []
    for xi in x:
        y.append(-2 * xi + np.random.normal(0, spread / 3) + bias)
    y = np.array(y)
    return x, y


def distance(x0, y0, k, b):
    return abs(-k*x0 + y0 - b) / (k**2 + 1)**0.5


def formula_regularization(train, trainRes, M, koefOfReg=0):
    # M- count of parameters of model
    # N- len(train)
    def getBasisFunc(degree):
        return lambda x: x**degree
    basisFunctions = [getBasisFunc(i) for i in range(M)]
    FI = lambda x: [func(x) for func in basisFunctions]
    matrixOfPlan = [FI(x) for x in train]
    matrixOfPlan = np.reshape(matrixOfPlan, (len(train), M))
    I = np.array([[int(i == j) for j in range(len(matrixOfPlan[0]))] for i in range(len(matrixOfPlan[0]))])
    I[0][0] = 0
    # we dont regularizate w0, because it's a bias, a bias maybe very big.
    # It's not a problem, cos our selection maybe be on 1000 above the 0X
    w = np.dot(
                np.dot(
                        np.linalg.inv(np.dot(matrixOfPlan.transpose(), matrixOfPlan) + I * koefOfReg),
                        matrixOfPlan.transpose()),
                trainRes)
    print(np.int32(w))
    return w, FI


def getModel1(train, trainRes, M, koefOfReg=0):
    w, FI = formula_regularization(train, trainRes, M)
    loss = 0
    for i in range(len(train)):
        loss += (trainRes[i] - np.dot(w.transpose(), FI(train[i])))**2
    loss = loss / 2
    print(M, "loss", loss)
    print()
    return lambda x: np.dot(w.transpose(), FI(x))


def grad_descent_with_my_func_loss(x, y):
    # cost func is func of distance
    k = 0
    b = 0
    epochs = 10000
    N = len(x)

    error = 0
    learning_rate = 10
    previous_error = 2**64
    for epoch in range(epochs):
        nablaK = 0
        nablaB = 0
        error = 0
        for xi, yi in zip(x, y):
            error += distance(xi, yi, k, b) ** 2
            nablaK += ((-k*xi + yi - b) / (abs(-k*xi + yi - b)) * -xi * np.sqrt(k**2 + 1) - abs(k * xi - yi + b) * k / np.sqrt(k**2 + 1)) / (k**2 + 1)
            nablaB += -1 / np.sqrt(k**2 + 1) * (-k*xi + yi - b) / (abs(-k*xi + yi - b))

        error /= N
        if previous_error < error and abs(error - previous_error) > 1:
            learning_rate /= 10
            print("CHANGE lr, epoch =", epoch, "because", previous_error, error,
                  abs(error - previous_error), "\n")
        if abs(previous_error - error) < 0.0000001 and error < 1:
            print(epoch, "stop because it's stoped")
            break
        k -= (nablaK / N) * learning_rate
        b -= (nablaB / N) * learning_rate
        previous_error = error
    print("error", error)
    return k, b


def parse_data(path):
    f = open(path)
    xs = np.array([])
    ys = np.array([])
    for line in f.readlines():
        l = line.split(",")
        xs = np.append(xs, l[:-1])
        ys = np.append(ys, l[-1])
    xs, ys = xs.astype(float), ys.astype(float)
    xs = xs.reshape((len(xs) // len(l[:-1]), len(l[:-1])))
    return xs, ys


def loss_RMSE(ys_predict, ys):
    loss = np.power(ys_predict - ys, 2)
    loss = np.sum(loss)
    loss /= len(ys)
    return loss


def surface_of_loss(theta0, theta1, x, y):
    surf = np.zeros((len(theta0), len(theta1)))
    for i, t0 in enumerate(theta0):
        for j, t1 in enumerate(theta1):
            y_pred = t0 + t1 * x
            surf[i, j] = loss_RMSE(y_pred, y)
    return surf


def get_nabla(x, y_pred, y, count_params):
    grad = np.zeros(count_params)
    grad[0] = (y_pred - y).sum()
    grad[1:] = ((y_pred - y) * x).sum(axis=0)
    grad /= x.shape[0]
    return grad


def get_lin_reg(x, y, epochs, learning_rate, count_params):
    error = 0
    theta = np.random.random(count_params)
    for epoch in range(epochs):
        y_pred = theta[0] + (x * theta[1:]).sum(axis=1)
        y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
        error = loss_RMSE(y_pred, y)
        nabla = get_nabla(x, y_pred, y, count_params)
        theta -= nabla * learning_rate
        if epoch % 250 == 0:
            print(f"№{epoch} loss={np.round(error, 3)}"
                  f" nabla={np.round(nabla, 4)}"
                  f" theta={np.round(theta, 4)}")
    print("error", error)
    return lambda x_: np.sum(theta[0] + theta[1:] * x_)


def plot_lin_reg_model(x, y, model):
    plot_x = np.arange(np.min(x), np.max(x), 0.1)
    plot_y = np.array([model(xi) for xi in plot_x]).reshape(-1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.reshape(-1), y=y.reshape(-1), mode='markers'))
    fig.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="lines"))
    fig.show()


def plot_lin_reg_model_with_2_features(x, y, model):
    fig = go.Figure()
    # size = x[:, 1] * std_x[1] + mean_x[1]
    fig.add_trace(go.Scatter3d(x=x[:, 0].reshape(-1), y=x[:, 1].reshape(-1), z=y.reshape(-1),
                               mode='markers'))

    plot_x0 = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    plot_x1 = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
    plot_y = np.zeros((len(plot_x0), len(plot_x1)))
    for i, xi0 in enumerate(plot_x0):
        for j, xj1 in enumerate(plot_x1):
            plot_y[i, j] = model(np.array([xi0, xj1]))
    fig.add_trace(go.Surface(x=plot_x0, y=plot_x1, z=plot_y))

    fig.show()


def plot_surface_of_error():
    surf_theta0 = np.arange(-10, 10, 0.1)
    surf_theta1 = np.arange(-10, 10, 0.1)
    surface = surface_of_loss(surf_theta0, surf_theta1, train_x, train_y)

    fig = plotly.subplots.make_subplots(rows=1, cols=2)
    fig.add_trace(go.Surface(x=surf_theta0, y=surf_theta1, z=surface))
    fig.add_trace(go.Contour(x=surf_theta0, y=surf_theta1, z=surface))
    fig.update_layout(title='Loss surface', autosize=False,
                      width=1980, height=1080)
    fig.show()


if __name__ == "__main__":
    # (340412.66, 110631.05, -6649.47)
    p = r"C:\Users\Norma\Downloads\files01\ex1data2.txt"
    train_x, train_y = parse_data(p)
    train_y = np.reshape(train_y, (train_y.shape[0], 1))
    print(train_x.shape, train_y.shape)
    mean_x = np.mean(train_x, axis=0)
    std_x = np.std(train_x, axis=0)
    norm_train_x = (train_x - mean_x) / std_x

    params = train_x.shape[1] + 1
    lin_reg = get_lin_reg(norm_train_x, train_y, epochs=1001, learning_rate=0.05,
                          count_params=params)

    # plot_lin_reg_model(train_x, train_y, lin_reg)
    plot_lin_reg_model_with_2_features(norm_train_x, train_y, lin_reg)
    # plot_surface_of_error()
