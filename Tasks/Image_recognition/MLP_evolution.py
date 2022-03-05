import random
from time import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, ConfusionMatrix

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MLP(torch.nn.Module):
    def __init__(self, count_of_inp_neu, count_of_out_neu):
        super().__init__()
        self.input = count_of_inp_neu
        self.output = count_of_out_neu
        self.linear1 = torch.nn.Linear(self.input, self.output, device=device)
        # self.linear2 = torch.zeros((self.hidden, self.output), device=device)

    def __call__(self, x):
        x = softmax(self.linear1(x))
        # x = self.softmax(torch.mm(x, self.linear2))
        return x

    def mutate(self, mean, std):
        shape_of_weight = self.linear1.weight.shape
        size_of_mutation = (1, 784)
        # size_of_mutation = (10, 784)
        indexes_of_mutation = (random.randint(0, shape_of_weight[0] - size_of_mutation[0]),
                               random.randint(0, shape_of_weight[1] - size_of_mutation[1]))
        self.linear1.weight[indexes_of_mutation[0]:indexes_of_mutation[0] + size_of_mutation[0],
                            indexes_of_mutation[1]:indexes_of_mutation[1] + size_of_mutation[1]] += \
            torch.normal(mean=mean, std=std, size=size_of_mutation, device=device)
        # self.linear2.weight += torch.normal(mean=self.mean, std=self.std,
        #                                     size=self.linear2.weight.shape, device=device)

    def clone(self):
        copy_net = MLP(self.input, self.output)
        copy_net.load_state_dict(self.state_dict())
        return copy_net


def get_gauss_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def median_kernel(mat, size, padding=0):
    pad_matrix = np.pad(mat, padding)
    res_a = np.zeros((pad_matrix.shape[0] - size//2 * 2, pad_matrix.shape[1] - size//2 * 2))
    for i_ in range(res_a.shape[0]):
        for j_ in range(res_a.shape[1]):
            res_a[i_, j_] = np.median(pad_matrix[i_:i_ + size, j_:j_ + size])
    return res_a


def convolution2d(mat, filter_kernel, padding=0):
    pad_matrix = np.pad(mat, padding)
    s = filter_kernel.shape + tuple(np.subtract(pad_matrix.shape, filter_kernel.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(pad_matrix, shape=s, strides=pad_matrix.strides * 2)
    return np.einsum('ij,ijkl->kl', filter_kernel, subM)


def parse_data(path):
    df = pd.read_csv(path, sep=",")
    y = df["label"]
    x = df.drop(columns="label")
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y


def get_data_loaders(x, y):
    percentOfSplit = 0.10

    x_train = x[:x.shape[0] - int(x.shape[0] * percentOfSplit), :]
    y_train = y[:y.shape[0] - int(y.shape[0] * percentOfSplit)]
    x_test = x[x.shape[0] - int(x.shape[0] * percentOfSplit):, :]
    y_test = y[y.shape[0] - int(y.shape[0] * percentOfSplit):]

    x_train = np.concatenate([x_train, x_train + np.random.normal(0, 0.3, size=x_train.shape)])
    y_train = np.concatenate([y_train, y_train])

    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, device=device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    return train_data_loader, test_data_loader


def fit(net: MLP, epochs, parameters, loss_func, train_dl, test_dl):
    global fig
    mean, std = parameters
    accuracy = Accuracy().to(device)
    start = time()
    x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]

    predict = net(x_test)
    loss = loss_func(predict, y_test)
    print("Null Loss", loss.item())
    for epoch in range(epochs):
        for x, y in train_dl:
            mutated_net = net.clone()
            mutated_net.mutate(mean, std)

            mut_predict = mutated_net(x)
            predict = net(x)
            mut_loss = loss_func(mut_predict, y)
            loss = loss_func(predict, y)
            # choose the bests
            if mut_loss < loss:
                net = mutated_net.clone()
        # save loss
        history_of_fittness.append(loss.cpu())
        # plotting graphs
        fig.add_trace(go.Scatter(x=list(range(len(history_of_fittness))), y=history_of_fittness),
                      row=4, col=1)
        row, col = 1, 1
        w = net.linear1.weight.detach().cpu().numpy()  # weight
        for digit in range(9):
            fig.add_heatmap(visible=False,
                            z=w[digit, :].reshape((28, 28)), row=row, col=col)
            col += 1
            if col == 4:
                col = 1
                row += 1
        # stat of fit
        if epoch % 10 == 0:
            predict_test = net(x_test)
            test_loss = loss_func(predict_test, y_test)
            acc = accuracy(predict_test, y_test).item()
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {round(test_loss.item(), 4)}. "
                  f"Acc = {round(acc, 3)}.")
            start = time()
    return net


def main():
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\train.csv"
    x, y = parse_data(path)
    x = x / np.std(x)
    x = x - np.mean(x)
    input_shape = x.shape[1]
    train_dl, test_dl = get_data_loaders(x, y)

    # loss_fn = Accuracy().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    print("START TRAINING!")
    epochs = 3*200 + 1
    mutation_parameters = (0, 0.003)  # mean and std
    null_net = MLP(input_shape, 10)
    with torch.no_grad():
        fitted_net = fit(null_net, epochs, mutation_parameters, loss_fn, train_dl, test_dl)

    steps = []
    for i in range(len(history_of_fittness)):
        step = dict(
            method='restyle',
            args=[{"visible": [False] * len(fig.data)}],
        )
        count_of_plots = 10
        for k in range(count_of_plots):
            step["args"][0]["visible"][count_of_plots * i + k] = True
        steps.append(step)

    sliders = [dict(steps=steps)]
    fig.layout.sliders = sliders

    x, y = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    y_predict = fitted_net(x)
    accuracy = Accuracy().to(device)
    acc = accuracy(y_predict, y)
    print(f"Result accuracy={round(acc.item(), 3)}")
    conf_mat = ConfusionMatrix(10).to(device)
    res_c_m = conf_mat(y_predict, test_dl.dataset.tensors[1])
    print(res_c_m)
    return fitted_net


if __name__ == "__main__":
    # https://towardsdatascience.com/evolving-neural-networks-b24517bb3701
    pio.renderers.default = "browser"
    softmax = torch.nn.Softmax(dim=-1)
    device = torch.device("cuda")
    # device = torch.device("cpu")
    history_of_fittness = []
    fig = make_subplots(rows=4, cols=3, subplot_titles=list(range(9)),
                        specs=[[{}, {}, {}],
                               [{}, {}, {}],
                               [{}, {}, {}],
                               [{"colspan": 3}, None, None]], )
    models = main()
    fig.show()
