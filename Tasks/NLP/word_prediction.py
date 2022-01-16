from time import time
import builtins
import torch.nn.functional as F
from torchmetrics import Precision, Recall
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch


def is_str_correct(text, alphabet):
    """checks if text is correct for us"""
    if len(text.split()) < 4:  # at least 3 words in sentence
        return False
    symbols = set(text)
    diff = symbols.difference(alphabet)
    if len(diff) != 0:
        return False
    return True


def preprocess_data(path) -> pd.Series:
    """returns preprocessed data"""
    # alphabet = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ' ")
    alphabet = set("aAaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz '")
    # alphabet.update('0123456789.,!?()-–—«» " ')
    data = pd.read_table(path, header=None, delimiter=r"\n", engine="python")
    data = data.iloc[:, 0]
    data = data.apply(builtins.str.lower)
    data = data.replace(regex=r"<\|startoftext\|>", value="")
    data = data.replace(regex=r"<\|endoftext\|>", value="")
    data = data.replace(regex=r"\xa0", value="")
    data = data.replace(regex=r"\ufeff", value="")
    data = data.replace(regex=r"…", value="")
    data = data.replace(regex='[-–—«»\d,!%:;"]', value=" ")
    data = data.replace(regex="[\*\(\)\?\.]", value=" ")
    indexes = data.apply(is_str_correct, args=(alphabet,))
    denied = data[~indexes]
    data = data[indexes]
    return data


def vocabulary_of_all_words(data):
    vocab = set()
    for row in data:
        vocab.update(row.split())
    vocab = list(vocab)
    word2ind_ = {x: ind for ind, x in enumerate(vocab)}
    ind2word_ = {ind: x for ind, x in enumerate(vocab)}
    return word2ind_, ind2word_


class NLPerceptron(torch.nn.Module):
    def __init__(self, count_of_inp_neurons):
        super().__init__()
        self.count_of_classes = count_of_inp_neurons
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=0)

        self.linear1 = torch.nn.Linear(count_of_inp_neurons, 250)
        # self.linear2 = torch.nn.Linear(100, 100)
        # self.linear3 = torch.nn.Linear(500, 500)
        self.linear2 = torch.nn.Linear(250, count_of_inp_neurons)

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        # x = self.sigmoid(self.linear2(x))
        # x = self.sigmoid(self.linear3(x))
        # x = self.softmax(self.linear4(x))
        x = self.softmax(self.linear2(x))
        return x

    def get_count_of_classes(self):
        return self.count_of_classes


def get_data_loaders(data, word2ind, width_of_context, step):
    """
    :param data:
    :param word2ind:
    :param width_of_context: is count of words we're given in net, to predict center word
    :param step: how big our step, when we take slices of original sentence
    """
    numeric_data = []
    max_len_of_row = 0
    for row in data:
        numeric_data.append(list(map(lambda x: word2ind[x], row.split())))
        max_len_of_row = max(max_len_of_row, len(row))
    # multiply because of one sentence we can make more than 1 input samples
    # it is just upper estimation
    # input_dataset = np.zeros((len(numeric_data)*7, len(word2ind)), dtype=np.float32)
    estimated_size = len(numeric_data) * (max_len_of_row // step)
    print("Upper estimation of size of dataset", estimated_size)
    input_dataset = np.zeros((estimated_size, width_of_context), dtype=np.long)
    output_dataset = np.zeros(estimated_size, dtype=np.long)
    s_time = time()
    index = 0
    for sentence in numeric_data:
        if len(sentence) <= width_of_context + 1:
            continue
            # slices = [(0, len(sentence))]
        else:
            slices = [(i, i + width_of_context + 1)
                      for i in range(0, len(sentence) - width_of_context - 1, step)]
        for slc in slices:
            beg, end = slc
            center = (beg + end) // 2
            output_dataset[index] = sentence[center]
            # input_dataset[index, sentence[beg:end]] = 1
            # input_dataset[index, sentence[center]] = 0
            input_dataset[index] = sentence[beg:center] + sentence[center+1:end]  # concatenation
            index += 1
    input_dataset = input_dataset[:index, :]
    output_dataset = output_dataset[:index]
    # output_dataset = np.reshape(output_dataset, (output_dataset.shape[0], 1))
    print(f"Shapes of datasets:\tinp: {input_dataset.shape}, out: {output_dataset.shape}")

    percentOfSplit = 0.15
    x_train = input_dataset[:input_dataset.shape[0] - int(input_dataset.shape[0] * percentOfSplit), :]
    y_train = output_dataset[:output_dataset.shape[0] - int(output_dataset.shape[0] * percentOfSplit)]
    x_test = input_dataset[input_dataset.shape[0] - int(input_dataset.shape[0] * percentOfSplit):, :]
    y_test = output_dataset[output_dataset.shape[0] - int(output_dataset.shape[0] * percentOfSplit):]

    train_inp_tensor = torch.tensor(x_train, dtype=torch.long, device=device)
    train_out_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    test_inp_tensor = torch.tensor(x_test, dtype=torch.long, device=device)
    test_out_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    train_dataset = TensorDataset(train_inp_tensor, train_out_tensor)
    test_dataset = TensorDataset(test_inp_tensor, test_out_tensor)

    train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=512)
    return train_data_loader, test_data_loader


def get_ohe_tensor(inp_x, count_of_classes):
    """  Tensor of indexes converts to tensor of One-Hot-Encoding
     shape=(batch_size, count_of_values)         shape=(batch_size, count_of_classes)
    [ [1561, 9876, 3243, 103],                  [ [0, ..., 0, 1, 0, ... 0],
                ...                converts to               ...
      [658, 3216, 752, 648] ]                     [0, ..., 0, 1, 0, ... 0] ]
    """
    out_x = torch.zeros((inp_x.shape[0], count_of_classes), device=device)
    row_indexes = torch.tensor(range(0, out_x.shape[0]), dtype=torch.long)
    row_indexes = torch.repeat_interleave(row_indexes, inp_x.shape[1])
    column_indexes = torch.flatten(inp_x)
    out_x[row_indexes, column_indexes] = 1
    return out_x


def similarity(word, matrix):
    index_of_word = word2ind[word]
    arr = torch.zeros((matrix.shape[0]), dtype=torch.float32)
    for ind, row in enumerate(matrix):
        arr[ind] = ((matrix[index_of_word] - row) ** 2).sum()
    indexes = torch.argsort(torch.Tensor(arr), descending=True)
    sim_words = torch.arange(0, len(matrix))[indexes]
    print(word, end=":\n")
    for i, ind in enumerate(sim_words[:15]):
        print(ind2word[ind.data.item()], end=" ")
        if (i+1) % 6 == 0:
            print()
    print()
    return sim_words


def fit(model, epochs, loss_func, opt, train, test):
    print("START TRAINING!")
    start_time = time()
    count_of_classes = model.get_count_of_classes()
    precision = Precision(num_classes=count_of_classes).to(device=device)
    recall = Recall(num_classes=count_of_classes).to(device=device)
    for epoch in range(epochs):
        for raw_x, y in train:
            x = get_ohe_tensor(raw_x, count_of_classes)
            y_predict = model(x)
            loss = loss_func(y_predict, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch < 5:
            print(f"Epoch {epoch} calc in {round(time() - start_time)}sec and loss = {loss.item()}")
            start_time = time()
    with torch.no_grad():
        # single evaluation, cos test DL without batches
        prec, rec = 0, 0
        for raw_x, y in test:
            # raw_x, y = test.dataset.tensors
            x = get_ohe_tensor(raw_x, count_of_classes)
            y_predict = model(x)
            # y = torch.nn.functional.one_hot(y, count_of_classes)
            prec += precision(y_predict, y)
            rec += recall(y_predict, y)
        prec, rec = prec / test_dl.batch_size, rec / test_dl.batch_size
        print(f"In the end:\n"
              f"\tprecision = {prec}\n"
              f"\trecall = {rec}")
    return model


if __name__ == "__main__":
    # todo use api of Kaggle
    # todo optimisation of backprop
    device = torch.device("cuda:0")
    # quotes = preprocess_data(
    #     r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\Amazing quotes\tagged.txt")
    # aneks, denied_aneks = preprocess_data(
    #     r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\anekdots\anek.txt"
    # )
    reviews = preprocess_data(
    r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\text reviews of films\reviews.txt")
    data = reviews[:20000]
    print("Count of quotes:", data.shape, end="\t")
    word2ind, ind2word = vocabulary_of_all_words(data)
    print("Count of unique words:", len(word2ind))
    context = 4
    train_dl, test_dl = get_data_loaders(data, word2ind, context, step=10)

    net = NLPerceptron(len(word2ind))
    net = net.to(device=device, dtype=torch.float32)
    loss_fn = torch.nn.NLLLoss(reduction="sum")

    learning_rate = 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    epochs = 1
    fitted_net = fit(net, epochs, loss_fn, optimizer, train_dl, test_dl)

    # print(similarity(word2ind["хороший"], net.linear2.weight.data))
    similarity("good", net.linear2.weight.data)
    similarity("bad", net.linear2.weight.data)
    similarity("terrible", net.linear2.weight.data)
    similarity("draw", net.linear2.weight.data)
    similarity("go", net.linear2.weight.data)
    similarity("film", net.linear2.weight.data)
    similarity("stick", net.linear2.weight.data)
    similarity("sample", net.linear2.weight.data)
    # print(np.argmax(fitted_net(train_dl.dataset[0][0]).cpu().detach().numpy()))
    # print(np.argmax(train_dl.dataset[0][1].cpu().detach().numpy()))
