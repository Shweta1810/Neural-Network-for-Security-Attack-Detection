"""From-scratch NumPy neural network: input -> sigmoid hidden -> softmax output.
Trained with L2-regularised mini-batch gradient descent on cross-entropy loss."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class NeuralNet:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        a1 = sigmoid(X @ self.W1 + self.b1)
        return a1, softmax(a1 @ self.W2 + self.b2)

    def predict_proba(self, X):
        return self.forward(X)[1]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def loss(self, X, y, lam=0.0):
        p = self.predict_proba(X)
        ce = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        return ce + 0.5 * lam * ((self.W1 ** 2).sum() + (self.W2 ** 2).sum())

    def fit(self, X, y, epochs=10, lr=0.5, batch_size=1000, lam=1e-4, seed=0, on_epoch=None):
        rng = np.random.default_rng(seed)
        n, k = len(X), self.b2.shape[0]
        for ep in range(1, epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                b = order[s:s + batch_size]
                Xb, m = X[b], len(b)
                a1, p = self.forward(Xb)
                d2 = p
                d2[np.arange(m), y[b]] -= 1.0
                d2 /= m
                dW2 = a1.T @ d2 + lam * self.W2
                db2 = d2.sum(0)
                d1 = (d2 @ self.W2.T) * a1 * (1 - a1)
                dW1 = Xb.T @ d1 + lam * self.W1
                db1 = d1.sum(0)
                self.W2 -= lr * dW2; self.b2 -= lr * db2
                self.W1 -= lr * dW1; self.b1 -= lr * db1
            if on_epoch:
                on_epoch(ep, self)
        return self
