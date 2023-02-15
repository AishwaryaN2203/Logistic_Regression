
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression() : 

    def __init__ (self, learningRate = 0.001, n_iterations = 1000):
        self.learningRate = learningRate
        self.n_iteration = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.n_iteration):
            linear_predictions = np.dot(X, self.weights) + self.bias
            prediction = sigmoid(linear_predictions)

            dw = (1/n_samples) * np.dot(X.T, (prediction - y))
            db = (1/n_samples) * np.sum(prediction-y)

            self.weights = self.weights - self.learningRate * dw
            self.bias = self.bias - self.learningRate*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred