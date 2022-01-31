from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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

    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    return train_data_loader, test_data_loader


def fit(model, epochs, loss_func, opt, train_dl, test_dl):
    start_time = time()
    accuracy = Accuracy().to(device)
    for epoch in range(epochs):
        for x, y in train_dl:
            y_predict = model(x)
            loss = loss_func(y_predict, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch < 5:
            print(f"Epoch {epoch} calc in {round(time() - start_time)}sec and loss = {loss.item()}")
            start_time = time()
    x, y = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    y_predict = model(x)
    acc = accuracy(y_predict, y)
    print(f"Result accuracy={round(acc.item(), 3)}")
    return model



def main():
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\train.csv"
    x, y = parse_data(path)
    x, y = x.to(device), y.to(device)
    std_ = torch.std(x)
    x = x / std_
    mean_ = torch.mean(x)
    x = x - mean_
    input_shape = x.shape[1]
    train_dl, test_dl = get_data_loaders(x, y)

    net = ConvNet()
    net = net.to(device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    print("START TRAINING!")
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    epochs = 7
    net = fit(net, epochs, loss_fn, optimizer, train_dl, test_dl)
    return net, mean_, std_


if __name__ == "__main__":
    device = torch.device("cuda")
    model, mean, std = main()
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\test.csv"
    df = pd.read_csv(path, sep=",")
    test_x = torch.tensor(df.to_numpy().reshape((df.shape[0], 1, 28, 28)),
                          device=torch.device("cpu"), dtype=torch.float32)
    test_x = test_x / std.to(torch.device("cpu"))
    test_x = test_x - mean.to(torch.device("cpu"))
    model = model.to(torch.device("cpu"))
    res_test = model(test_x)

    res_test = torch.argmax(res_test, dim=1)
    df_res_test = pd.DataFrame(res_test.detach().cpu().numpy())
    df_res_test = df_res_test.reset_index()
    df_res_test.iloc[:, 0] += 1
    df_res_test.to_csv(r"D:\res_mnist.csv", header=["ImageId", "Label"], index=False)
