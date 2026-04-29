"""
ANN Titanic Training
Usage: python ann_titanic_train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_preprocess import preprocess_titanic


def main():
    print("=" * 50)
    print("ANN Titanic Train")
    print("=" * 50)

    train_path = "../data/titanic/titanic_train.csv"
    test_path = "../data/titanic/titanic_test.csv"
    model_dir = "../models"
    result_dir = "../results"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train, X_test, y_test, mean, std = preprocess_titanic(train_df, test_df)
    print(f"    Train samples: {X_train.shape[0]}")
    print(f"    Test samples:  {X_test.shape[0]}")
    print(f"    Features: {X_train.shape[1]}")

    # 2. Train ANN
    print("\n[2] Training ANN (MLPClassifier)...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    model.fit(X_train, y_train)
    print(f"    Iterations: {model.n_iter_}")
    print(f"    Final loss: {model.loss_:.6f}")

    # 3. Evaluate
    print("\n[3] Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Test accuracy: {acc * 100:.2f}%")

    # 4. Plot loss curve
    print("\n[4] Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ANN Titanic - Loss Curve (Training)')
    plt.legend()
    plt.grid(alpha=0.3)
    loss_path = os.path.join(result_dir, "ann_titanic_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Loss curve saved to: {loss_path}")

    # 5. Save model + preprocessing params
    print("\n[5] Saving model...")
    model_data = {
        "model": model,
        "mean": mean,
        "std": std,
        "accuracy": acc
    }
    model_path = os.path.join(model_dir, "ann_titanic.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to: {model_path}")

    # 6. Save result
    result_path = os.path.join(result_dir, "accuracy.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"ANN Titanic Test Accuracy: {acc * 100:.2f}%\n")
    print(f"\n[6] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
