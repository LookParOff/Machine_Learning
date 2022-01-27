from time import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy
# https://towardsdatascience.com/evolving-neural-networks-b24517bb3701


class EvolutionNet:
    def __init__(self, count_of_inp_neu, param):
        self.input = count_of_inp_neu
        self.mean, self.std = param
        self.hidden = 50
        self.output = 10
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        # specific initialisation for activation relu
        self.linear1 = 0.2 * torch.rand((self.input, self.hidden), device=device) - 0.1
        self.linear2 = 0.2 * torch.rand((self.hidden, self.output), device=device) - 0.1

    def mutate(self):
        self.linear1 += torch.normal(mean=self.mean, std=self.std,
                                     size=self.linear1.shape, device=device)
        self.linear2 += torch.normal(mean=self.mean, std=self.std,
                                     size=self.linear2.shape, device=device)
        # self.linear1 *= (2 * torch.rand(size=self.linear1.shape, device=device)) - 1
        # self.linear2 *= (2 * torch.rand(size=self.linear2.shape, device=device)) - 1

    def __call__(self, x):
        x = self.relu(torch.mm(x, self.linear1))
        x = self.softmax(torch.mm(x, self.linear2))
        return x

    def __deepcopy__(self):
        copy_net = EvolutionNet(self.input, (self.mean, self.std))
        copy_net.linear1 = self.linear1.__deepcopy__({})
        copy_net.linear2 = self.linear2.__deepcopy__({})
        return copy_net


def parse_data(path):
    df = pd.read_csv(path, sep=",")
    y = df["label"]
    x = df.drop(columns="label")
    x = torch.tensor(x.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y.to_numpy())
    return x, y


def get_data_loaders(x, y):
    percentOfSplit = 0.10
    x_train = x[:x.shape[0] - int(x.shape[0] * percentOfSplit), :]
    y_train = y[:y.shape[0] - int(y.shape[0] * percentOfSplit)]
    x_test = x[x.shape[0] - int(x.shape[0] * percentOfSplit):, :]
    y_test = y[y.shape[0] - int(y.shape[0] * percentOfSplit):]

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    return train_data_loader, test_data_loader


def fit(population: np.array, epochs, loss_func, train_dl, test_dl, descending=False):
    # if we need minimize loss_func- descending should be True
    print(train_dl.dataset.tensors[1].shape, test_dl.dataset.tensors[1].shape)
    num_of_bests = 10  # how many of the best species do we take from the population
    best_species = None  # the best species of the population
    result = torch.zeros(len(population), dtype=torch.float32)
    accuracy = Accuracy().to(device)
    start = time()
    for epoch in range(epochs):
        for x, y in train_dl:
            for id_specie, specie in enumerate(population):
                y_predict = specie(x)
                loss = loss_func(y_predict, y)
                result[id_specie] = loss.item()
            best_ids = torch.argsort(result, descending=descending)[:num_of_bests]  # chosen the bests
            best_species = population[best_ids]
            for id_specie, specie in enumerate(population):
                population[id_specie] = best_species[id_specie % num_of_bests].__deepcopy__()
                if id_specie > num_of_bests:
                    population[id_specie].mutate()
        if epoch % 10 == 0:
            x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
            y_predict = best_species[0](x_test)
            loss = round(loss_func(y_predict, y_test).item(), 5)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {loss}. "
                  f"Acc = {acc}")
            start = time()
    return best_species


def main():
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\train.csv"
    x, y = parse_data(path)
    x, y = x.to(device), y.to(device)
    x = x / torch.std(x)
    x = x - torch.mean(x)
    input_shape = x.shape[1]
    train_dl, test_dl = get_data_loaders(x, y)

    # loss_fn = Accuracy().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    print("START TRAINING!")
    epochs = 70 + 1
    len_of_population = 100
    mutation_parameters = (0, 0.3)  # mean and std
    population = np.array([EvolutionNet(input_shape, mutation_parameters)
                           for _ in range(len_of_population)])
    arr_of_nets = fit(population, epochs, loss_fn, train_dl, test_dl)

    fitted_net = arr_of_nets[0]
    x, y = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    y_predict = fitted_net(x)
    accuracy = Accuracy().to(device)
    acc = accuracy(y_predict, y)
    print(f"Result accuracy={round(acc.item(), 3)}")
    return arr_of_nets


if __name__ == "__main__":
    device = torch.device("cuda")
    # device = torch.device("cpu")
    model = main()
