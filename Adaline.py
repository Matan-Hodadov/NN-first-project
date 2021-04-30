import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VECTOR_SIZE = 1000
TRAIN_TEST_SPLIT = int(VECTOR_SIZE * .7)


class CustomAdaline(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.weights = None
        self.score_ = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        gets random weights and optimizes them n_iterations times
        :param X: input data to train on
        :param y: input labels to optimize the model with
        :return:
        """
        # use random seed to get initial value for weights
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # go over the rows applying the weights and optimizing them
        for _ in range(self.n_iterations):
            for row in range(X.shape[0]):
                activation_function_output = self.net_input(X[row])
                error = y[row] - activation_function_output
                self.weights[1:] = self.weights[1:] + self.learning_rate * error
                self.weights[0] = self.weights[0] + self.learning_rate * error
        print("the weights are: " + str(self.weights))

    def net_input(self, X: np.array) -> np.array:
        """
        apply the weights and bias on the input data
        :param X: input data
        :return: probabilities for possible label
        """
        X_probs = np.dot(X, self.weights[1:]) + self.weights[0]
        return X_probs

    def predict(self, X: np.array) -> np.array:
        """
        predicts labels using fitted algorithm
        :param X: input data
        :return: labels given by algorithm
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def score(self, y_true: np.array, y_pred: np.array) -> float:
        """
        calculate the accuracy on predicted values by the algorithm
        :param y_true: how the labels should be
        :param y_pred: labels predicted by algorithm
        :return: calced score
        """
        self.score_ = np.sum(y_true == y_pred) / len(y_true) * 100
        return self.score_

    @staticmethod
    def prepare_data(resolution: int, size: int = VECTOR_SIZE, random_state: int = 1, labeling: int = 0) -> np.array:
        """
        create the data with custom query, resolution and size
        :param resolution: the fraction size of the data (100 is 1/100)
        :param size:
        :param random_state:
        :param labeling: if 0 then all data with x > 1/2 & y >1/2 label 1, if 1 then 1/2  <= x**2 + y**2 <= 3/4 label 1
        :return:
        """
        data = np.zeros(shape=(size, 3))
        np.random.seed(random_state)
        data[:, :2] = np.random.randint(-resolution, resolution + 1, size=(size, 2)) / resolution

        if labeling == 0:
            for row in data:
                row[2] = 1 if row[0] > 0.5 and row[1] > 0.5 else -1
        if labeling == 1:
            for row in data:
                row[2] = 1 if 1 / 2 <= row[0] ** 2 + row[1] ** 2 <= 3 / 4 else -1

        return data

    def train_test_split(self, data: np.ndarray, split: int = TRAIN_TEST_SPLIT)-> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        X = data[:, :2]
        y = data[:, 2]

        X_train = X[:split]
        X_test = X[split:]
        y_train = y[:split]
        y_test = y[split:]

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("part A")
    adaline = CustomAdaline(random_state=1)
    data = adaline.prepare_data(resolution=100, random_state=1)

    X_train, X_test, y_train, y_test = adaline.train_test_split(data)

    adaline.fit(X_train, y_train)
    y_pred = adaline.predict(X_test)
    score = adaline.score(y_test, y_pred)
    print(score)

    print("Q1: does differnt train data change the accuracy?")
    data = adaline.prepare_data(resolution=100, random_state=33)
    X_train2, _, y_train2, _ = adaline.train_test_split(data)
    print("is the new data the same? " + str((X_train2 == X_train).all()))
    adaline.fit(X_train2, y_train2)
    y_pred = adaline.predict(X_test)
    score = adaline.score(y_test, y_pred)
    print(score)

    print("Q2: how does more data and precision effect the accuracy?")
    VECTOR_SIZE = 10000
    data = adaline.prepare_data(resolution=10000, size=VECTOR_SIZE, random_state=1)
    X_train, X_test, y_train, y_test = adaline.train_test_split(data)
    adaline.fit(X_train, y_train)
    y_pred = adaline.predict(X_test)
    score = adaline.score(y_test, y_pred)
    print(score)

    print("\npart B")
    print("Q1: change the labels to be 1 if 1/2  <= x**2 + y**2 <= 3/4")
    VECTOR_SIZE = 1000
    data = adaline.prepare_data(resolution=100, size=VECTOR_SIZE, random_state=1, labeling=1)
    X_train, X_test, y_train, y_test = adaline.train_test_split(data)
    adaline.fit(X_train, y_train)
    y_pred = adaline.predict(X_test)
    score = adaline.score(y_test, y_pred)
    print(score)

    print("Q2: check if adding more data helps")
    scores = np.zeros(10)
    data_size = np.zeros(10)
    for i in range(10):
        data = adaline.prepare_data(resolution=100, size=500*(1+i), random_state=1, labeling=1)
        X_train, X_test, y_train, y_test = adaline.train_test_split(data)
        adaline.fit(X_train, y_train)
        y_pred = adaline.predict(X_test)
        score = adaline.score(y_test, y_pred)
        scores[i] = score
        data_size[i] = 500*(1+i)

    plt.plot(data_size, scores)
    plt.show()