import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VECTOR_SIZE = 1000
TRAIN_TEST_SPLIT = int(VECTOR_SIZE*.7)


class CustomAdaline(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.coef_ = None
        self.score_ = None

    def fit(self, X: np.array, y: np.array) -> None:
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            activation_function_output = self.activation_function(self.net_input(X))
            errors = y - activation_function_output
            self.coef_[1:] = self.coef_[1:] + self.learning_rate * X.T.dot(errors)
            self.coef_[0] = self.coef_[0] + self.learning_rate * errors.sum()

    def net_input(self, X: np.array) -> np.array:
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    @staticmethod
    def activation_function(X: np.array) -> np.array:
        return X

    def predict(self, X: np.array) -> np.array:
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)

    # def score(self, X: np.array, y: np.array) -> np.array:
    #     misclassified_data_count = 0
    #     for xi, target in zip(X, y):
    #         output = self.predict(xi)
    #         if target != output:
    #             misclassified_data_count += 1
    #     total_data_count = len(X)
    #     self.score_ = (total_data_count - misclassified_data_count) / total_data_count
    #     return self.score_

    def score(self, y_true: np.array, y_pred: np.array) -> float:
        self.score_ = np.sum(y_true == y_pred) / len(y_true)
        return self.score_

    @staticmethod
    def prepare_data(resolution: int) -> np.array:
        data = np.zeros(shape=(VECTOR_SIZE, 3))
        data[:, :2] = np.random.randint(-resolution, resolution + 1, size=(VECTOR_SIZE, 2)) / resolution

        for row in data:
            row[2] = 1 if row[0] > 0.5 and row[1] > 0.5 else -1

        return data


if __name__ == '__main__':
    print("here")
    adaline = CustomAdaline()
    data = adaline.prepare_data(100)

    X = data[:, :2]
    y = data[:, 2]

    X_train = X[:TRAIN_TEST_SPLIT]
    X_test = X[TRAIN_TEST_SPLIT:]
    y_train = y[:TRAIN_TEST_SPLIT]
    y_test = y[TRAIN_TEST_SPLIT:]

    adaline.fit(X_train, y_train)
    y_pred = adaline.predict(X_test)
    scores = adaline.score(y_test, y_pred)
    print(scores)
