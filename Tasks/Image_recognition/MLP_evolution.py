import os
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
        self.linear1 = self.linear1.requires_grad_(False)
        # self.linear2 = torch.zeros((self.hidden, self.output), device=device)

    def __call__(self, x):
        x = softmax(self.linear1(x))
        # x = self.softmax(torch.mm(x, self.linear2))
        return x

    def mutate(self, mean, std):
        shape_of_weight = self.linear1.weight.shape
        size_of_mutation = (1, 784)
        # size_of_mutation = (10, 784)
        idx_of_mutation = (random.randint(0, shape_of_weight[0] - size_of_mutation[0]),
                           random.randint(0, shape_of_weight[1] - size_of_mutation[1]))
        mutated_net = self.clone()
        mutated_net.linear1.weight[idx_of_mutation[0]:idx_of_mutation[0] + size_of_mutation[0],
                                   idx_of_mutation[1]:idx_of_mutation[1] + size_of_mutation[1]] += \
            torch.normal(mean=mean, std=std, size=size_of_mutation, device=device)
        # self.linear2.weight += torch.normal(mean=self.mean, std=self.std,
        #                                     size=self.linear2.weight.shape, device=device)
        return mutated_net

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
    limit_size_population = len(population)
    result = torch.zeros(len(population), dtype=torch.float32)
    accuracy = Accuracy().to(device)
    start = time()
    for epoch in range(epochs):
        for x, y in train_dl:
            offspring_population = np.array([])
            prob = np.random.random()  # prob of crossover
            if prob > 0:
                # mutation
                for id_specie, specie in enumerate(population):
                    offspring_population = np.append(offspring_population, specie)
                    for _ in range(3):
                        offspring_population = np.append(offspring_population,
                                                         specie.mutate(mean, std))
            else:
                # crossover
                while len(offspring_population) < limit_size_population:
                    id_mother, id_father = torch.randint(10, size=(2,))
                    if id_father == id_mother:
                        continue
                    mother = population[id_mother]
                    father = population[id_father]
                    offspring = father.clone()
                    weight_shape = offspring.linear1.weight.shape
                    # generate random indexes, to give something to offspring from mother
                    # but not more than a half
                    count_crossover = weight_shape[0] * weight_shape[1] // 2
                    indexes_dim_0 = torch.randint(0, weight_shape[0], size=(count_crossover, 1))
                    indexes_dim_1 = torch.randint(0, weight_shape[1], size=(count_crossover, 1))

                    offspring.linear1.weight.data[indexes_dim_0, indexes_dim_1] = \
                        mother.linear1.weight.data[indexes_dim_0, indexes_dim_1]
                    offspring_population = np.append(offspring_population, offspring)
            population = np.append(population, offspring_population)

            # keep the bests
            result = torch.zeros(len(population), dtype=torch.float32)
            for id_specie, specie in enumerate(population):
                y_predict = specie(x)
                loss = loss_func(y_predict, y)
                result[id_specie] = loss
            best_ids = torch.argsort(result, descending=descending)
            population = population[best_ids]
            population = population[:limit_size_population]
            result = result[best_ids]

        # save loss
        history_of_fittness.append(result[0])
        # plotting graphs
        fig.add_trace(go.Scatter(x=list(range(len(history_of_fittness))), y=history_of_fittness),
                      row=4, col=1)
        row, col = 1, 1
        w = population[0].linear1.weight.detach().cpu().numpy()  # weight
        for digit in range(9):
            fig.add_heatmap(visible=False,
                            z=w[digit, :].reshape((28, 28)), row=row, col=col)
            col += 1
            if col == 4:
                col = 1
                row += 1
        # stat of fit
        if epoch % 5 == 0:
            x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
            y_predict = population[0](x_test)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {round(torch.min(result).item(), 4)}. "
                  f"Acc = {acc}.")
            start = time()
    return population


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\train.csv"
    x, y = parse_data(path)
    x = x / np.std(x)
    x = x - np.mean(x)
    input_shape = x.shape[1]
    train_dl, test_dl = get_data_loaders(x, y)

    # loss_fn = Accuracy().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    print("START TRAINING!")
    epochs = 100 + 1
    len_of_population = 10
    mutation_parameters = (0, 0.003)  # mean and std
    population = np.array([MLP(input_shape, 10)
                           for _ in range(len_of_population)])
    with torch.no_grad():
        arr_of_nets = fit(population, epochs, mutation_parameters, loss_fn, train_dl, test_dl)

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

    fitted_net = arr_of_nets[0]
    x, y = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    y_predict = fitted_net(x)
    accuracy = Accuracy().to(device)
    acc = accuracy(y_predict, y)
    print(f"Result accuracy={round(acc.item(), 3)}")
    conf_mat = ConfusionMatrix(10).to(device)
    res_c_m = conf_mat(y_predict, test_dl.dataset.tensors[1])
    print(res_c_m)
    return arr_of_nets


if __name__ == "__main__":
    # https://towardsdatascience.com/evolving-neural-networks-b24517bb3701
    pio.renderers.default = "browser"
    softmax = torch.nn.Softmax(dim=-1)
    # device = torch.device("cuda")
    device = torch.device("cpu")
    history_of_fittness = []
    fig = make_subplots(rows=4, cols=3, subplot_titles=list(range(9)),
                        specs=[[{}, {}, {}],
                               [{}, {}, {}],
                               [{}, {}, {}],
                               [{"colspan": 3}, None, None]], )
    models = main()
    fig.show()
