import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path)
    X = df[["x1", "x2", "x3", "x4"]].values.astype(float)
    y = df["y"].values.astype(float)
    return X, y


def zscore_scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def minmax_scale(X):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    X_scaled = (X - x_min) / denom
    return X_scaled, x_min, denom


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def compute_loss(X, y, beta):
    pred = X @ beta
    return np.mean((pred - y) ** 2)


def least_squares(X, y):
    """
    最小二乘闭式解:
    beta = (X^T X)^(-1) X^T y
    为了更稳定，这里用伪逆
    """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta


def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    beta = np.zeros(X.shape[1])
    loss_history = []
    n = X.shape[0]

    for epoch in range(epochs):
        pred = X @ beta
        grad = (2 / n) * (X.T @ (pred - y))
        beta = beta - lr * grad
        loss_history.append(compute_loss(X, y, beta))

    return beta, loss_history


def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    beta = np.zeros(X.shape[1])
    n = X.shape[0]
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
            pred_i = xi @ beta
            grad = 2 * (pred_i - yi) * xi
            beta = beta - lr * grad

        loss_history.append(compute_loss(X, y, beta))

    return beta, loss_history


def mini_batch_gradient_descent(X, y, lr=0.01, epochs=200, batch_size=16):
    beta = np.zeros(X.shape[1])
    n = X.shape[0]
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]

            pred_b = xb @ beta
            grad = (2 / len(xb)) * (xb.T @ (pred_b - yb))
            beta = beta - lr * grad

        loss_history.append(compute_loss(X, y, beta))

    return beta, loss_history


def predict(X, beta):
    return X @ beta


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def plot_loss(loss_dict, save_path=None):
    plt.figure(figsize=(8, 6))
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_prediction(y_true, y_pred, title="Prediction", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title(title)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


class LinearRegressionModel:
    """
    method:
        - 'least_squares'
        - 'bgd'
        - 'sgd'
        - 'mini_batch'
    """

    def __init__(
        self,
        method="least_squares",
        lr=0.01,
        epochs=1000,
        batch_size=16,
        scale_method="zscore"
    ):
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scale_method = scale_method

        self.beta = None
        self.loss_history = []
        self.scale_param_1 = None
        self.scale_param_2 = None

    def _fit_scaler(self, X):
        if self.scale_method == "zscore":
            X_scaled, p1, p2 = zscore_scale(X)
        elif self.scale_method == "minmax":
            X_scaled, p1, p2 = minmax_scale(X)
        else:
            raise ValueError("scale_method must be 'zscore' or 'minmax'")

        self.scale_param_1 = p1
        self.scale_param_2 = p2
        return X_scaled

    def _transform(self, X):
        if self.scale_method == "zscore":
            return (X - self.scale_param_1) / self.scale_param_2
        elif self.scale_method == "minmax":
            return (X - self.scale_param_1) / self.scale_param_2
        else:
            raise ValueError("scale_method must be 'zscore' or 'minmax'")

    def fit(self, X, y):
        X_scaled = self._fit_scaler(X)
        X_bias = add_bias(X_scaled)

        if self.method == "least_squares":
            self.beta = least_squares(X_bias, y)
            self.loss_history = [compute_loss(X_bias, y, self.beta)]

        elif self.method == "bgd":
            self.beta, self.loss_history = batch_gradient_descent(
                X_bias, y, lr=self.lr, epochs=self.epochs
            )

        elif self.method == "sgd":
            self.beta, self.loss_history = stochastic_gradient_descent(
                X_bias, y, lr=self.lr, epochs=self.epochs
            )

        elif self.method == "mini_batch":
            self.beta, self.loss_history = mini_batch_gradient_descent(
                X_bias, y, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size
            )

        else:
            raise ValueError("method must be one of: least_squares, bgd, sgd, mini_batch")

    def predict(self, X):
        X_scaled = self._transform(X)
        X_bias = add_bias(X_scaled)
        return predict(X_bias, self.beta)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        }

    def save(self, file_path):
        data = {
            "method": self.method,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "scale_method": self.scale_method,
            "beta": self.beta,
            "loss_history": self.loss_history,
            "scale_param_1": self.scale_param_1,
            "scale_param_2": self.scale_param_2
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def load(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.method = data["method"]
        self.lr = data["lr"]
        self.epochs = data["epochs"]
        self.batch_size = data["batch_size"]
        self.scale_method = data["scale_method"]
        self.beta = data["beta"]
        self.loss_history = data["loss_history"]
        self.scale_param_1 = data["scale_param_1"]
        self.scale_param_2 = data["scale_param_2"]


def run_demo(
    data_path="./house_data.csv",
    method="least_squares",
    lr=0.01,
    epochs=1000,
    batch_size=16,
    scale_method="zscore",
    results_dir="../results"
):
    os.makedirs(results_dir, exist_ok=True)

    X, y = load_data(data_path)

    model = LinearRegressionModel(
        method=method,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scale_method=scale_method
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = model.evaluate(X, y)

    print("=" * 50)
    print(f"Method: {method}")
    print(f"Beta: {model.beta}")
    print(f"MSE:  {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")

    if method != "least_squares":
        plot_loss(
            {method.upper(): model.loss_history},
            save_path=os.path.join(results_dir, f"{method}_loss.png")
        )

    plot_prediction(
        y,
        y_pred,
        title=f"{method.upper()} Prediction",
        save_path=os.path.join(results_dir, f"{method}_prediction.png")
    )

    model.save(os.path.join(results_dir, f"{method}_linear_model.pkl"))


if __name__ == "__main__":
    # 你可以在这里切换不同方法测试
    run_demo(
        data_path="./house_data.csv",
        method="mini_batch",   # least_squares / bgd / sgd / mini_batch
        lr=0.01,
        epochs=500,
        batch_size=16,
        scale_method="zscore"
    )
