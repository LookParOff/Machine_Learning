from time import time
import builtins
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
    """returns preprocessed and cleaned data"""
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
    estimated_size = len(numeric_data) * (max_len_of_row // step)
    print("Upper estimation of size of dataset", estimated_size)
    input_dataset = np.zeros((estimated_size, width_of_context), dtype=np.long)
    output_dataset = np.zeros(estimated_size, dtype=np.long)
    index = 0
    for sentence in numeric_data:
        if len(sentence) <= width_of_context + 1:
            # skip of small sentence
            continue
        slices = [(i, i + width_of_context + 1)
                  for i in range(0, len(sentence) - width_of_context - 1, step)]
        for slc in slices:
            beg, end = slc
            center = (beg + end) // 2
            if len(set(sentence[beg:center] + sentence[center+1:end])) \
                    != len(sentence[beg:center] + sentence[center+1:end]):
                continue  # denied slices with repeatable words
            output_dataset[index] = sentence[center]
            input_dataset[index] = sentence[beg:center] + sentence[center+1:end]  # concatenation
            index += 1
    input_dataset = input_dataset[:index, :]
    output_dataset = output_dataset[:index]
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


class NLPerceptron(torch.nn.Module):
    """
    Word2Vec. bag of word Predict center word on context
    """
    def __init__(self, count_of_inp_neurons):
        """
        each neuron in inputs neurons corresponding with some word.
        So neuron on index 42 represent word with key 42 in dict ind2word
        """
        super().__init__()
        self.count_of_classes = count_of_inp_neurons
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=0)

        self.linear1 = torch.nn.Linear(count_of_inp_neurons, 50)
        # self.linear2 = torch.nn.Linear(100, 100)
        # self.linear3 = torch.nn.Linear(500, 500)
        self.linear2 = torch.nn.Linear(50, count_of_inp_neurons)

    def forward(self, batch_of_x):
        """
        batch_of_x.shape = (batch_size, vocabulary_size)
         in each row list of indexes of words, where neurons = 1. This is representation of sentence

        next took weights and summed them for each x in batch_of_x,
         this is effectively, rather simple multiplication.
        """
        layer1 = self.linear1.weight.T[batch_of_x]
        layer1 = torch.sum(layer1, dim=1)
        layer1 = self.sigmoid(layer1)
        x = self.softmax(self.linear2(layer1))
        return x

    def get_count_of_classes(self):
        return self.count_of_classes


def similarity(word, matrix):
    """
    Find the most similar words by trained weights of model
    """
    index_of_word = word2ind[word]
    arr = torch.zeros((matrix.shape[0]), dtype=torch.float32)
    for i, row in enumerate(matrix):
        arr[i] = ((matrix[index_of_word] - row) ** 2).sum()
    indexes = torch.argsort(torch.Tensor(arr))
    arr = arr[indexes]
    sim_words = torch.arange(0, len(matrix))[indexes]
    print(word, end=":\n")
    for i, ind_of_word in enumerate(sim_words[:7]):
        print("\t", ind2word[ind_of_word.data.item()], arr[i].data.item())
    print()
    return sim_words


def fit(model, epochs, loss_func, opt, train):
    """
    return trained model
    """
    start_time = time()
    for epoch in range(epochs):
        for raw_x, y in train:
            y_predict = model(raw_x)
            loss = loss_func(y_predict, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch < 1:
            print(f"Epoch {epoch} calc in {round(time() - start_time)}sec and loss = {loss.item()}")
            start_time = time()
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # quotes = preprocess_data(
    #     r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\Amazing quotes\tagged.txt")
    # aneks, denied_aneks = preprocess_data(
    #     r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\anekdots\anek.txt"
    # )
    reviews = preprocess_data(
    r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\text reviews of films\reviews.txt")
    data = reviews[:3000]
    print("Count of quotes:", data.shape, end="\t")
    word2ind, ind2word = vocabulary_of_all_words(data)
    print("Count of unique words:", len(word2ind))
    context = 4
    # skip test_dl because we check model by finding similar words by function similarity()
    train_dl, _ = get_data_loaders(data, word2ind, context, step=1)

    net = NLPerceptron(len(word2ind))
    net = net.to(device=device, dtype=torch.float32)
    loss_fn = torch.nn.NLLLoss(reduction="sum")
    name = input("Please input the name of this model\t")
    path_of_save = r"C:\Users\Norma\PycharmProjects\Machine Learning\Saved_models"
    print("START TRAINING!")
    learning_rate = 0.01
    for _ in range(3):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        epochs = 16
        net = fit(net, epochs, loss_fn, optimizer, train_dl)
        learning_rate = learning_rate / 100
        torch.save(net.state_dict(), fr"{path_of_save}\{name}.pt")
    similarity("good", net.linear1.weight.T.data)
    similarity("bad", net.linear1.weight.T.data)
    similarity("man", net.linear1.weight.T.data)
    similarity("woman", net.linear1.weight.T.data)
