"""
ANN for House Price Regression
Usage: python ann_house.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    X = df[["x1", "x2", "x3", "x4"]].values.astype(float)
    y = df["y"].values.astype(float)
    return X, y


def main():
    print("=" * 50)
    print("ANN House Price Regression")
    print("=" * 50)

    data_path = "../data/house/house_data.csv"
    model_dir = "../models"
    result_dir = "../results"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load data
    print("\n[1] Loading data...")
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train samples: {X_train.shape[0]}")
    print(f"    Test samples:  {X_test.shape[0]}")

    # 2. Standardize
    print("\n[2] Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Train ANN
    print("\n[3] Training ANN (MLPRegressor)...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    model.fit(X_train, y_train)
    print(f"    Iterations: {model.n_iter_}")
    print(f"    Final loss: {model.loss_:.6f}")

    # 4. Evaluate
    print("\n[4] Evaluating...")
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"    MSE:  {mse:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    MAE:  {mae:.6f}")

    # 5. Plot loss curve
    print("\n[5] Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Training Loss', linewidth=2)
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        plt.plot(model.validation_scores_, label='Validation Score', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ANN House Price - Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    loss_path = os.path.join(result_dir, "ann_house_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Loss curve saved to: {loss_path}")

    # 6. Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('ANN House Price - Prediction vs Actual')
    plt.legend()
    plt.grid(alpha=0.3)
    pred_path = os.path.join(result_dir, "ann_house_prediction.png")
    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Prediction plot saved to: {pred_path}")

    # 7. Save model
    print("\n[6] Saving model...")
    model_data = {
        "model": model,
        "scaler": scaler,
        "metrics": {"MSE": mse, "RMSE": rmse, "MAE": mae}
    }
    model_path = os.path.join(model_dir, "ann_house.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to: {model_path}")

    # 8. Save result
    result_path = os.path.join(result_dir, "accuracy.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"ANN House - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}\n")
    print(f"\n[7] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
