import torch


class Polynom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(10, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.power = torch.concat([torch.ones(5, dtype=torch.float64),
                                   torch.ones(5, dtype=torch.float64) + 1])

    def __call__(self, x):
        xx = torch.concat([x, x], dim=1)
        return torch.mul(self.w, torch.pow(xx, self.power)).sum(dim=1) + self.b


# torch.set_printoptions(precision=6, sci_mode=False)
file_input = open("input.txt", "r")
train_x = []
train_y = []
for _ in range(1000):
    row = file_input.readline().split()
    row = list(map(lambda x: float(x), row))
    train_x.append(row[:-1])
    train_y.append(row[-1])

test_x = []
for _ in range(1000):
    row = file_input.readline().split()
    row = list(map(lambda x: float(x), row))
    test_x.append(row)

train_x, train_y, test_x = torch.tensor(train_x, dtype=torch.float64), \
                           torch.tensor(train_y, dtype=torch.float64), \
                           torch.tensor(test_x, dtype=torch.float64)

# data_x = torch.rand((2000, 5), dtype=torch.float64) * 10 - 5
# data_x = data_x[torch.randperm(2000)]
# data_y = 12 * torch.pow(data_x[:, 0], 2) + 1 * data_x[:, 1] + 0.01 * data_x[:, 2]\
#           - 4.8 * data_x[:, 3] + 5 * data_x[:, 4] + 10
# train_x = data_x[:1000]
# train_y = data_y[:1000]
# test_x = data_x[1000:]
# test_y = data_y[1000:]
mean_x = torch.mean(train_x, dim=0)
std_x = torch.std(train_x, dim=0)
train_x = (train_x - mean_x) / std_x

model = Polynom()

learning_rate = 1
batch_size = 250
step_count = 500 + 1

loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [], 0.1)

for t in range(step_count):
    for id_batch in range(batch_size, len(train_x) + 1, batch_size):
        y_pred = model(train_x[id_batch - batch_size:id_batch])
        loss = loss_fn(y_pred, train_y[id_batch - batch_size:id_batch])
        optim.zero_grad()
        loss.backward()
        optim.step()
    # if t % 100 == 0:
    #     predict_y = model((test_x - mean_x) / std_x)
    #     temp = torch.abs(predict_y - test_y)
    #     acc = torch.count_nonzero(temp < 1e-6).item() / 1000
    #     print(t, "Accuracy 1e-6:", acc)
    #     if acc > 0.98:
    #         break
    # scheduler.step()

# print("koefs of power 1", model.w.data[:5], "\n")
# print("koefs of power 2", model.w.data[5:], "\n")
# print("bias", model.b, "\n")

predict_y = model((test_x - mean_x) / std_x)
# predict_y = model(test_x)
# temp = torch.abs(predict_y - test_y)
# print("Accuracy with precision 1:", torch.count_nonzero(temp < 1).item() / 1000)
# print("Accuracy with precision 2:", torch.count_nonzero(temp < 1e-2).item() / 1000)
# print("Accuracy with precision 4:", torch.count_nonzero(temp < 1e-4).item() / 1000)
# print("Accuracy with precision 6:", torch.count_nonzero(temp < 1e-6).item() / 1000)


for el in predict_y:
    print(el.item())
