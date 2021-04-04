import numpy as np
from sklearn import datasets


class LinearRegression(object):
    def __init__(self, num_iter=10000, lr=1e-2):
        self.num_iter = num_iter
        self.lr = lr

    def init_weights(self, dim):
        limit = 1 / np.sqrt(dim)
        self.w = np.random.normal(0, limit, (dim, 1))

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def train(self, X, y):
        # add bias column
        X = np.insert(X, 0, values=1, axis=1)
        n, d = X.shape
        self.init_weights(d)
        for epoch in range(self.num_iter):
            y_pred = self.predict(X)
            loss = self.loss(y, y_pred)
            if epoch % 5 == 0:
                print(f"Loss at epoch {epoch} is {loss}")
            self.update(X, y, y_pred, self.lr)

    def update(self, X, y, y_pred, lr):
        n, d = X.shape
        grad_w = (X.T @ (y_pred - y)) / n
        self.w = self.w - lr * grad_w

    def predict(self, X):
        return X @ self.w


if __name__ == "__main__":
    # get data first
    X, y = datasets.make_regression(n_features=10, n_informative=4)
    model = LinearRegression(num_iter=100000, lr=0.001)
    model.train(X, y)
