import numpy as np
import pandas as pd
from lipo_loader import load_lipo


def relu(x):
    return x * (x > 0)


def h(x):
    return 1.0 * (x > 0)


class TwoLayerPerceptron:
    def __init__(self, hidden_size, eps, sigma):
        self.hidden_size = hidden_size
        self.eps = eps
        self.sigma = sigma

        try:
            df = pd.read_csv("lipo.csv")
        except FileNotFoundError:
            exit(404)

        del df["Вещество"]
        self.min_values = df.min()
        self.max_values = df.max()
        ndf = (df - self.min_values) / (self.max_values - self.min_values)
        self.y = ndf["logP"].values.reshape(-1, 1)
        self.x = ndf.iloc[:, 1:].values
        self.W1 = np.random.randn(self.hidden_size, len(self.x[0]))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1)
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = relu(z2)
        return a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # Number of training examples

        # Backward pass
        dz2 = self.a2 - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update parameters
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def train(self):
        flag = True
        while flag:
            z1 = [np.dot(self.W1,i) + self.b1 for i in self.x]
            a1 = [relu(i) for i in z1]
            z2 = [np.dot(i, self.W2) + self.b2 for i in a1]
            L = [(self.y[i] - z2[i])[0] for i in range(len(self.x))]
            V = [
                np.outer(self.W2, h(np.dot(self.W1,self.x[i]) + self.b1))[0]
                for i in range(len(self.x))
            ]
            db2j = np.sum(L)
            dw2j = np.sum(
                [
                    np.outer(
                        np.dot(L[i], relu(np.dot(self.W1,self.x[i]) + self.b1)), 1
                    )
                    for i in range(len(self.x))
                ],
                axis=0,
            )
            db1j = np.sum(
                [np.outer(V[i] * L[i], 1) for i in range(len(self.x))], axis=0
            )
            dw1j = np.sum(
                [np.outer(V[i] * L[i], self.x[i]) for i in range(len(self.x))], axis=0
            )
            dj = [db2j, dw2j, db1j, dw1j]
            for i in dj:
                try:
                    if np.linalg.norm(i) < self.sigma:
                        return
                except ValueError:
                    pass
            self.b2 += db2j * self.eps
            np.add(self.W2, dw2j * self.eps)
            np.add(self.b1, db1j * self.eps)
            np.add(self.W1, dw1j * self.eps)

    def predict(self, X):
        # Make predictions
        return self.forward(X)


if __name__ == "__main__":
    model = TwoLayerPerceptron(hidden_size=32, eps=0.1, sigma=0.1)
    model.train()
    for i in range(len(model.x)):
        print(f"{model.predict(model.x[i]).flatten()}\t{model.y[i]}")
