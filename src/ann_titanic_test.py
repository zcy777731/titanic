"""
ANN Titanic Testing
Usage: python ann_titanic_test.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data_preprocess import preprocess_titanic


def main():
    print("=" * 50)
    print("ANN Titanic Test")
    print("=" * 50)

    train_path = "../data/titanic/titanic_train.csv"
    test_path = "../data/titanic/titanic_test.csv"
    model_path = "../models/ann_titanic.pkl"
    result_dir = "../results"
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load model
    print("\n[1] Loading model...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    print(f"    Model loaded (accuracy on train: {model_data.get('accuracy', 'N/A')})")

    # 2. Load and preprocess data
    print("\n[2] Loading and preprocessing data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train, X_test, y_test, mean, std = preprocess_titanic(train_df, test_df)

    # 3. Predict
    print("\n[3] Predicting...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Test accuracy: {acc * 100:.2f}%")

    # 4. Plot loss curve from saved model (if available)
    print("\n[4] Plotting model loss curve...")
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Training Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('ANN Titanic - Loss Curve (Test)')
        plt.legend()
        plt.grid(alpha=0.3)
        loss_path = os.path.join(result_dir, "ann_titanic_loss_test.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Loss curve saved to: {loss_path}")
    else:
        print("    No loss curve available in model.")

    # 5. Save result
    result_path = os.path.join(result_dir, "accuracy.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"ANN Titanic Test (loaded model): {acc * 100:.2f}%\n")
    print(f"\n[5] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
