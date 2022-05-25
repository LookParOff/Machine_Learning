from time import time
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


pio.renderers.default = "browser"


class MLP(torch.nn.Module):
    def __init__(self, count_of_inp_neu):
        super().__init__()
        self.input_shape = count_of_inp_neu
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.linear1 = torch.nn.Linear(self.input_shape, 10, device=device)
        # self.linear2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = self.softmax(self.linear1(x))
        return x


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

    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    return train_data_loader, test_data_loader


def fit(model, epochs, loss_func, opt, train_dl, test_dl):
    start_time = time()
    accuracy = Accuracy().to(device)
    x_test, y_test = test_dl.dataset.tensors[0], test_dl.dataset.tensors[1]
    for epoch in range(epochs):
        for x, y in train_dl:
            y_predict = model(x)
            loss = loss_func(y_predict, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        history_of_fittness.append(loss.item())
        fig.add_trace(go.Scatter(x=list(range(len(history_of_fittness))), y=history_of_fittness),
                      row=4, col=1)
        row, col = 1, 1
        w = model.linear1.weight.detach().cpu().numpy()  # weight
        for digit in range(9):
            fig.add_heatmap(visible=False,
                            z=w[digit, :].reshape((28, 28)), row=row, col=col)
            col += 1
            if col == 4:
                col = 1
                row += 1

        if epoch % 10 == 0:
            y_predict = model(x_test)
            acc = round(accuracy(y_predict, y_test).item(), 3)
            print(f"Epoch {epoch} calc in {round(time() - start_time)}sec. "
                  f"Loss = {history_of_fittness[-1]}. Test acc = {acc}")
            start_time = time()

    y_predict = model(x_test)
    acc = accuracy(y_predict, y_test)
    print(f"Result accuracy={round(acc.item(), 3)}")
    return model


def main():
    path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\train.csv"
    x, y = parse_data(path)
    x, y = x.to(device), y.to(device)
    x = x / torch.std(x)
    x = x - torch.mean(x)
    input_shape = x.shape[1]
    train_dl, test_dl = get_data_loaders(x, y)

    net = MLP(input_shape)
    net = net.to(device=device, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    print("START TRAINING!")
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    epochs = 100 + 1
    net = fit(net, epochs, loss_fn, optimizer, train_dl, test_dl)
    return net


if __name__ == "__main__":
    device = torch.device("cuda")
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

    sliders = [dict(steps=steps)]
    fig.layout.sliders = sliders
    fig.show()
    out_file = open("stat classic 0.001 101 epoch.txt", "w")
    h = []
    for el in history_of_fittness:
        h.append(str(el) + "\n")
    out_file.writelines(h)
    out_file.close()
    # path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\MNIST\test.csv"
    # df = pd.read_csv(path, sep=",")
    # test_x = torch.tensor(df.to_numpy(), dtype=torch.float32, device=device)
    # res_test = model(test_x)
    # res_test = torch.argmax(res_test, dim=1)
    # df_res_test = pd.DataFrame(res_test.detach().cpu().numpy())
    # df_res_test.iloc[:, 0] += 1
    # df_res_test.to_csv(r"D:\res_mnist.csv", header=["ImageId", "Label"], index=False)
