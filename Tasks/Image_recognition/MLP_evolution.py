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
        self.linear1.weight += torch.normal(mean=mean, std=std,
                                            size=self.linear1.weight.shape, device=device)
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


def fit(population: np.array, epochs, parameters, loss_func, train_dl, test_dl, descending=False):
    global fig
    """if we need minimize loss_func- descending should be True"""
    # how many of the best species do we take from the population
    mean, std = parameters
    num_of_bests = int(len(population) * 0.1)
    best_species = None  # the best species of the population
    result = torch.zeros(len(population), dtype=torch.float32)
    accuracy = Accuracy().to(device)
    start = time()

    # blur_kernel = get_gauss_kernel(3, 0.3)
    for epoch in range(epochs):
        if epoch % 5 == 0:
            for specie in population:
                blured_weight = median_kernel(specie.linear1.weight.cpu().numpy(), 3, 1)
                blured_weight = torch.tensor(blured_weight, dtype=torch.float32, device=device)
                specie.linear1.weight = torch.nn.Parameter(blured_weight)
        for x, y in train_dl:
            for id_specie, specie in enumerate(population):
                y_predict = specie(x)
                loss = loss_func(y_predict, y)
                result[id_specie] = loss
            # choose the bests
            best_ids = torch.argsort(result, descending=descending)
            best_species = population[best_ids[:num_of_bests]]
            result = result[best_ids]
            for id_specie, specie in enumerate(population):
                population[id_specie] = best_species[id_specie % num_of_bests].clone()
                if id_specie > num_of_bests:
                    population[id_specie].mutate(mean, std)

        # save loss
        history_of_fittness.append(result[0].item())
        # plotting graphs
        fig.add_trace(go.Scatter(x=list(range(len(history_of_fittness))), y=history_of_fittness),
                      row=4, col=1)
        row, col = 1, 1
        w = population[0].linear1.weight.detach().cpu().numpy()  # weight
        for i in range(9):
            fig.add_heatmap(visible=False,
                            z=w[i, :].reshape((28, 28)), row=row, col=col)
            col += 1
            if col == 4:
                col = 1
                row += 1
        if epoch % 10 == 0:
            x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
            y_predict = best_species[0](x_test)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {round(torch.min(result).item(), 4)}. "
                  f"Acc = {acc}")
            start = time()
    return best_species


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
    epochs = 50 + 1
    len_of_population = 20
    mutation_parameters = (0, 0.001)  # mean and std
    population = np.array([MLP(input_shape, 10)
                           for _ in range(len_of_population)])
    with torch.no_grad():
        arr_of_nets = fit(population, epochs, mutation_parameters, loss_fn, train_dl, test_dl)

    fitted_net = arr_of_nets[0]
    x, y = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    y_predict = fitted_net(x)
    accuracy = Accuracy().to(device)
    acc = accuracy(y_predict, y)
    print(f"Result accuracy={round(acc.item(), 3)}")
    return arr_of_nets


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
    model = main()

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

    sliders = [dict(
        steps=steps,
    )]

    fig.layout.sliders = sliders
    fig.show()
