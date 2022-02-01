from time import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy


class ConvEvolution(torch.nn.Module):
    def __init__(self, mutation_param):
        super().__init__()
        self.mean, self.std = mutation_param
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.linear1 = torch.nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = x.reshape(x.size(0), -1)
        x = self.softmax(self.linear1(x))
        return x

    def mutate(self):
        self.conv1.weight += torch.normal(mean=self.mean, std=self.std,
                                          size=self.conv1.weight.shape, device=device)
        # self.conv2.weight += torch.normal(mean=self.mean, std=self.std,
        #                                   size=self.conv2.weight.shape, device=device)
        self.linear1.weight += torch.normal(mean=self.mean, std=self.std,
                                            size=self.linear1.weight.shape, device=device)
        # self.linear2.weight += torch.normal(mean=self.mean, std=self.std,
        #                                     size=self.linear2.weight.shape, device=device)

    def clone(self):
        copy_net = ConvEvolution((self.mean, self.std)).to(device=device)
        # copy_net.conv1.weight = self.conv1.weight.clone()
        # copy_net.conv2.weight = self.conv2.weight.clone()
        # copy_net.linear1.weight = self.linear1.weight.clone()
        # copy_net.linear2.weight = self.linear2.weight.clone()

        # copy_net.conv1 = self.conv1
        # copy_net.conv2 = self.conv2
        # copy_net.linear1 = self.linear1
        # copy_net.linear2 = self.linear2

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
    percentOfSplit = 0.10
    x_train = x[:x.shape[0] - int(x.shape[0] * percentOfSplit), :]
    y_train = y[:y.shape[0] - int(y.shape[0] * percentOfSplit)]
    x_test = x[x.shape[0] - int(x.shape[0] * percentOfSplit):, :]
    y_test = y[y.shape[0] - int(y.shape[0] * percentOfSplit):]

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    return train_data_loader, test_data_loader


def fit(population: np.array, epochs, loss_func, train_dl, test_dl, descending=False):
    """if we need minimize loss_func- descending should be True"""
    # how many of the best species do we take from the population
    num_of_bests = int(len(population) * 0.1)
    best_species = None  # the best species of the population
    result = torch.zeros(len(population), dtype=torch.float32)
    accuracy = Accuracy().to(device)
    start = time()
    for epoch in range(epochs):
        for x, y in train_dl:
            for id_specie, specie in enumerate(population):
                y_predict = specie(x)
                loss = loss_func(y_predict, y)
                result[id_specie] = loss
            best_ids = torch.argsort(result, descending=descending)[:num_of_bests]  # chosen the bests
            best_species = population[best_ids]
            for id_specie, specie in enumerate(population):
                population[id_specie] = best_species[id_specie % num_of_bests].clone()
                if id_specie > num_of_bests:
                    population[id_specie].mutate()
        if epoch % 1 == 0:
            x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
            y_predict = best_species[0](x_test)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {round(torch.min(result).item(), 4)}. "
                  f"Acc = {acc}")
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
        net = fit(population, epochs, loss_fn, train_dl, test_dl)
    return net, mean_sample, std_sample


if __name__ == "__main__":
    device = torch.device("cuda")
    model, mean_, std_ = main()
