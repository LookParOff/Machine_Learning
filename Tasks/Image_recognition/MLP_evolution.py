from time import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt


class MLP(torch.nn.Module):
    def __init__(self, count_of_inp_neu, param):
        super().__init__()
        self.input = count_of_inp_neu
        self.mean, self.std = param
        self.hidden = 50
        self.output = 10
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.linear1 = torch.nn.Linear(self.input, 10, device=device)
        # self.linear2 = torch.zeros((self.hidden, self.output), device=device)

    def __call__(self, x):
        x = self.softmax(self.linear1(x))
        # x = self.softmax(torch.mm(x, self.linear2))
        return x

    def mutate(self):
        self.linear1.weight += torch.normal(mean=self.mean, std=self.std,
                                            size=self.linear1.weight.shape, device=device)
        # self.linear2.weight += torch.normal(mean=self.mean, std=self.std,
        #                                     size=self.linear2.weight.shape, device=device)

    def clone(self):
        copy_net = MLP(self.input, (self.mean, self.std))
        copy_net.load_state_dict(self.state_dict())
        return copy_net


class NetEnsemble:
    def __init__(self, count_of_inp_neu, param):
        self.input = count_of_inp_neu
        self.mean, self.std = param
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.linear1 = torch.zeros((self.input, 10), device=device)


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
            best_ids = torch.argsort(result, descending=descending)[
                       :num_of_bests]  # chosen the bests
            best_species = population[best_ids]
            for id_specie, specie in enumerate(population):
                population[id_specie] = best_species[id_specie % num_of_bests].clone()
                if id_specie > num_of_bests:
                    population[id_specie].mutate()
        if epoch % 10 == 0:
            x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
            y_predict = best_species[0](x_test)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            print(f"№{epoch} end in {round(time() - start)}secs. "
                  f"Loss = {round(torch.min(result).item(), 4)}. "
                  f"Acc = {acc}")
            fig, ax = plt.subplots(3, 3)
            row, col = 0, 0
            m = best_species[0].linear1.weight.detach().cpu().numpy()
            for i in range(9):
                sns.heatmap(m[i, :].reshape((28, 28)), ax=ax[row, col])
                col += 1
                if col == 3:
                    col = 0
                    row += 1
            plt.title(f"epoch №{epoch}", fontsize=20)
            plt.tight_layout()
            plt.show()
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
    epochs = 1 + 1
    len_of_population = 20
    mutation_parameters = (0, 0.001)  # mean and std
    population = np.array([MLP(input_shape, mutation_parameters)
                           for _ in range(len_of_population)])
    with torch.no_grad():
        arr_of_nets = fit(population, epochs, loss_fn, train_dl, test_dl)

    fitted_net = arr_of_nets[0]
    x, y = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    y_predict = fitted_net(x)
    accuracy = Accuracy().to(device)
    acc = accuracy(y_predict, y)
    print(f"Result accuracy={round(acc.item(), 3)}")
    conf_mat = ConfusionMatrix(10).to(device)
    conf_mat(y_predict, test_dl.dataset.tensors[1])
    return arr_of_nets


if __name__ == "__main__":
    # https://towardsdatascience.com/evolving-neural-networks-b24517bb3701
    device = torch.device("cuda")
    # device = torch.device("cpu")
    model = main()

    fig, ax = plt.subplots(3, 3)
    row, col = 0, 0
    for i in range(9):
        sns.heatmap(model[0].linear1.weight.detach().cpu().numpy()[i, :].reshape((28, 28)),
                    ax=ax[row, col])
        col += 1
        if col == 3:
            col = 0
            row += 1
    plt.title("END", fontsize=20)
    plt.tight_layout()
    plt.show()
