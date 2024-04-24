import numpy as np
from lipo_loader import load_lipo


class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        dz2 = self.a2 - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)


lipo_data = load_lipo()
X = np.array([i.parameters for i in lipo_data])
y = np.array([[i.logP] for i in lipo_data])

model = TwoLayerPerceptron(input_size=len(X[0]), hidden_size=4, output_size=1)
model.train(X, y, epochs=1000, learning_rate=0.1)
