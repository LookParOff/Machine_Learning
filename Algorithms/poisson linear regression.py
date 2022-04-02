import numpy as np
import matplotlib.pyplot as plt


def add_intercept(x):
    """Добавить столбец с коэффициентами для свободных членов.

    Аргументы:
        x: 2D NumPy array.

    Возвращаемое значение:
        Новая матрица, являющаяся конкатенацией столбца из единиц и матрицы x.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Загрузить датасет из CSV файла.

    Аргументы:
         csv_path: Пусть к CSV файлу с датасетом.
         label_col: Имя столбца, содержащего классы (должно быть 'y' или 'l').
         add_intercept: Добавить столбец из 1 к матрице x.

    Возвращаемое значение:
        xs: Numpy array со входными значениями x.
        ys: Numpy array с выходными значениями y.
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, correction=1.0):
    """Построить график датасета и решающей границы регрессии Пуассона.

    Args:
        x: Матрица обучающих примеров, по одному на строке.
        y: Вектор классов из {0, 1}.
        theta: Вектор параметров регрессионной модели.
        save_path: Имя файла для сохранения графика.
        correction: Коррекционный фактор.
    """
    # Построить график датасета
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Построить график решающей границы (найденной в результате решения уравнения theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    # plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    # plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    # Добавить метки и сохранить на диск.
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


class PoissonRegression:
    """Регрессия Пуассона.

    Пример использования:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=5000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Аргументы:
            step_size: Скорость обучения (learning rate).
            max_iter: Максимальное количество итераций.
            eps: Порог для определения сходимости.
            theta_0: Начальное значение theta. Если None, используется нулевой вектор.
            verbose: Печатать значения функции потерь во время обучения.
        """
        self.w = np.random.random((4, 1))
        self.b = np.random.random(1)
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def get_loss_RMSE(self, y, y_predict):
        loss = np.power(y_predict - y, 2)
        loss = np.sum(loss)
        loss /= len(y)
        return loss

    def fit(self, x, y):
        """Выполнить градиентный подъем для максимизации функции правдоподобия регрессии Пуассона.

        Аргументы:
            x: Обучающие примеры. Размерность (n_examples, dim).
            y: Классы. Размерность (n_examples,).
        """
        for i in range(self.max_iter):
            y_predict = self.predict(x)
            loss = self.get_loss_RMSE(y, y_predict)
            if self.verbose and i % 1000 == 0:
                print(loss)
            self.w = self.w + \
                     self.step_size * ((y - np.exp(y_predict)) * x).sum(axis=0, keepdims=True).T
            self.b = self.b + self.step_size * (y - np.exp(y_predict)).sum()
        # *** НАЧАЛО ВАШЕГО КОДА ***
        # *** КОНЕЦ ВАШЕГО КОДА ***

    def predict(self, x):
        """Выдать прогноз для значений x.

        Аргументы:
            x: Входные данные размерности (n_examples, dim).

        Возвращаемое значение:
            Вещественный прогноз для каждого входного значения, размерность (n_examples,).
        """
        y_predict = np.dot(x, self.w) + self.b
        return y_predict


def plot_task(y, y_predict, save_path="save_img.png"):
    plt.scatter(y, y_predict,  c='blue')
    plt.savefig(save_path)
    plt.show()


def main(lr, train_path, eval_path, save_path):
    """Задача: Регрессия Пуассона с градиентным подъемом.

    Аргументы:
        lr: Скорость обучения (learning rate) для градиентного подъема.
        train_path: Путь к CSV файлу, содержащему обучающую выборку.
        eval_path: Путь к CSV файлу, содержащему тестовую выборку.
        save_path: Путь к файлу для сохранения результата прогнозирования.
    """
    # Загружаем обучающую выборку
    x_train, y_train = load_dataset(train_path)
    x_test, y_test = load_dataset(eval_path)
    # mean_x = np.mean(x_train, axis=0)
    # std_x = np.std(x_train, axis=0)
    # x_train = (x_train - mean_x) / std_x
    # x_test = (x_test - mean_x) / std_x
    model = PoissonRegression(verbose=True, max_iter=3000)
    model.fit(x_train, np.reshape(y_train, (y_train.shape[0], 1)))

    print("train", y_test[:10])
    print("predict", model.predict(x_test)[:10])
    plot_task(y_test, model.predict(x_test))

    # theta = np.concatenate([[model.b], model.w])
    # plot(x_train, y_train, theta, "save_save_save_smth0.png")


if __name__ == '__main__':
    m_path = r"C:\Users\Norma\Downloads\TSU_ML\1.1 Регрессия Пуасона\ex1.1\\"
    main(lr=1e-5,
         train_path=m_path + 'train.csv',
         eval_path=m_path + 'valid.csv',
         save_path=m_path + 'poisson_pred.txt')
