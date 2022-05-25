from time import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy

import seaborn as sns
import matplotlib.pyplot as plt


class ConvEvolution(torch.nn.Module):
    def __init__(self, mutation_param):
        super().__init__()
        self.mean, self.std = mutation_param
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=2,
                                     device=device)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=2,
                                     device=device)
        self.linear1 = torch.nn.Linear(7 * 7 * 32, 1000, device=device)
        self.linear2 = torch.nn.Linear(1000, 10, device=device)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return x

    def mutate(self):
        self.conv1.weight += torch.normal(mean=self.mean, std=self.std,
                                          size=self.conv1.weight.shape, device=device)
        self.conv2.weight += torch.normal(mean=self.mean, std=self.std,
                                          size=self.conv2.weight.shape, device=device)
        self.linear1.weight += torch.normal(mean=self.mean, std=self.std,
                                            size=self.linear1.weight.shape, device=device)
        self.linear2.weight += torch.normal(mean=self.mean, std=self.std,
                                            size=self.linear2.weight.shape, device=device)

    def clone(self):
        copy_net = ConvEvolution((self.mean, self.std))
        copy_net.load_state_dict(self.state_dict())
        return copy_net


def parse_data(path):
    df = pd.read_csv(path, sep=",")
    y = df["label"]
    x = df.drop(columns="label")
    x = x.to_numpy()
    y = y.to_numpy()
    x = np.reshape(x, (x.shape[0], 28, 28))
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y)
    x = torch.unsqueeze(x, 1)  # one chanel in image
    return x, y


def get_data_loaders(x, y):
    percentOfSplit = 0.15
    x_train = x[:x.shape[0] - int(x.shape[0] * percentOfSplit), :]
    y_train = y[:y.shape[0] - int(y.shape[0] * percentOfSplit)]
    x_test = x[x.shape[0] - int(x.shape[0] * percentOfSplit):, :]
    y_test = y[y.shape[0] - int(y.shape[0] * percentOfSplit):]

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    return train_data_loader, test_data_loader


def fit(population: np.array, epochs, loss_func, train_dl, test_dl,
        w_decay=0.1, descending=False):
    """if we need minimize loss_func- descending should be True"""
    # how many of the best species do we take from the population
    num_of_bests = int(len(population) * 0.1)
    best_species = None  # the best species of the population
    result = torch.zeros(len(population), dtype=torch.float32)
    accuracy = Accuracy().to(device)
    start = time()
    x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    for epoch in range(epochs):
        for x, y in train_dl:
            for id_specie, specie in enumerate(population):
                y_predict = specie(x)
                loss = loss_func(y_predict, y)
                # for param in specie.parameters():
                #     loss += w_decay * (torch.pow(param, 2).sum())
                result[id_specie] = loss
            best_ids = torch.argsort(result, descending=descending)[:num_of_bests]  # chosen the bests
            best_species = population[best_ids]

            for id_specie, specie in enumerate(population):
                population[id_specie] = best_species[id_specie % num_of_bests].clone()
                if id_specie > num_of_bests:
                    population[id_specie].mutate()

        if epoch % 1 == 0:
            y_predict = best_species[0](x_test)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            best_loss_item = round(torch.min(result).item(), 4)

            loss_history.append(best_loss_item)
            acc_history.append(acc)
            num_epochs.append(epoch)
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {loss_history[-1]}. "
                  f"Acc = {acc_history[-1]}")
        start = time()
    return best_species


def main():
    torch.cuda.empty_cache()
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\train.csv"
    x, y = parse_data(path)
    x, y = x.to(device), y.to(device)
    std_sample = torch.std(x)
    x = x / std_sample
    mean_sample = torch.mean(x)
    x = x - mean_sample
    train_dl, test_dl = get_data_loaders(x, y)

    len_of_population = 21
    mutation_parameters = (0, 0.003)
    population = np.array([ConvEvolution(mutation_parameters).to(device)
                           for _ in range(len_of_population)])

    print("START TRAINING!")
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 75
    with torch.no_grad():
        net = fit(population, epochs, loss_fn, train_dl, test_dl, w_decay=0.000001)
    return net, mean_sample, std_sample


if __name__ == "__main__":
    loss_history = []
    acc_history = []
    num_epochs = []
    df = pd.DataFrame([loss_history, acc_history, num_epochs]).T
    df.columns = ["loss_history", "acc_history", "num_epochs"]

    device = torch.device("cuda")
    model, mean_, std_ = main()

    xs = [i for i in range(len(loss_history))]
    sns.set(rc={'figure.figsize': (15, 10)})
    fig, ax = plt.subplots(2, 1)
    sns.lineplot(x=xs, y=loss_history, ax=ax[0])  # , hue=num_epochs)
    sns.lineplot(x=xs, y=acc_history, ax=ax[1])  # , hue=num_epochs)
    fig.show()
