import numpy as np
from sklearn import datasets


class LogisticRegression(object):
    def __init__(self, num_iter=10000, lr=1e-1):
        self.num_iter = num_iter
        self.lr = lr

    def init_weights(self, dim):
        limit = 1 / np.sqrt(dim)
        self.w = np.random.uniform(-limit, limit, (dim, 1))

    def loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred + 1e-7) + (1 - y) * np.log(1 - y_pred + 1e-7))

    def train(self, X, y):
        # add bias column
        n, d = X.shape
        self.init_weights(d)
        for epoch in range(self.num_iter):
            y_pred = self.predict(X)
            loss = self.loss(y, y_pred)
            if epoch % 5 == 0:
                print(f"Loss at epoch {epoch} is {loss}")
            self.update(X, y, y_pred, self.lr)

    def update(self, X, y, y_pred, lr):
        grad_w = X.T @ (y_pred - y)
        self.w = self.w - lr * grad_w

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return self.sigmoid(X @ self.w)


def accuracy(y, y_pred):
    return 100 * np.mean(y == y_pred)


if __name__ == "__main__":
    # get data first
    np.random.seed(1)
    X, y = datasets.make_blobs(n_samples=1000, centers=2)
    # Adding bias column
    X = np.insert(X, 0, values=1, axis=1)
    y = np.expand_dims(y, axis=1)
    model = LogisticRegression(num_iter=1000, lr=0.1)
    model.train(X, y)
    y_pred = model.predict(X)
    print(f"Accuracy is {accuracy(y, y_pred)}%")
