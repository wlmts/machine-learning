import numpy as np

# Mean Squared Error Loss class for linear regression
class MSELoss:
    def forward(self, y_pred, y_true):
        n = len(y_true)
        return np.sum((y_pred - y_true) ** 2) / n

    def backward(self, y_pred, y_true, X):
        n = len(y_true)
        diff = y_pred - y_true
        grad_w = 2 * X.T @ diff / n
        grad_b = 2 * np.sum(diff) / n
        return grad_w, grad_b

# Gradient Descent Optimizer class
class GradientDescent:
    def __init__(self, lr):
        self.lr = lr

    def step(self, w, b, grad_w, grad_b):
        w = w - self.lr * grad_w
        b = b - self.lr * grad_b
        return w, b


if __name__ == "__main__":
    np.random.seed(42)
    # create simple data
    X = np.random.randn(100, 2)
    true_w, true_b = np.array([2, -3]), 4
    y = X @ true_w + true_b + 0.1*np.random.randn(100)

    w, b = np.zeros(2), 0.0
    loss_fn = MSELoss()
    opt = GradientDescent(lr=0.1)

    # training loop
    for epoch in range(500):
        y_pred = X @ w + b
        gw, gb = loss_fn.backward(y_pred, y, X)
        w, b = opt.step(w, b, gw, gb)
        if (epoch+1) % 100 == 0:
            print(f"epoch{epoch+1:3d} loss={loss_fn.forward(y_pred,y):.4f} w={w.round(2)} b={b:.2f}")

    print(f"\ntrue:w{true_w}, b={true_b}")
