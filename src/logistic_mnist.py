import os
import pickle
import time
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_mnist(data_dir, train_ratio=5/6):
    X_train, y_train = [], []
    X_test, y_test = [], []
    classes = sorted(os.listdir(data_dir))

    for label, digit in enumerate(classes):
        digit_dir = os.path.join(data_dir, digit)
        fnames = sorted(os.listdir(digit_dir))
        split_idx = int(len(fnames) * train_ratio)

        for fname in fnames[:split_idx]:
            img = Image.open(os.path.join(digit_dir, fname)).convert("L")
            X_train.append(np.array(img).flatten())
            y_train.append(label)

        for fname in fnames[split_idx:]:
            img = Image.open(os.path.join(digit_dir, fname)).convert("L")
            X_test.append(np.array(img).flatten())
            y_test.append(label)

    return (np.array(X_train, dtype=np.float32), np.array(y_train),
            np.array(X_test, dtype=np.float32), np.array(y_test))


def main():
    print("=" * 50)
    print("Logistic Regression MNIST")
    print("=" * 50)

    data_dir = "../data/mnist_images"
    model_path = "../models/logistic_mnist.pkl"
    os.makedirs("../models", exist_ok=True)

    # 1. Load data
    print("\n[1] Loading MNIST data...")
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    print(f"    Train samples: {X_train.shape[0]}")
    print(f"    Test samples:  {X_test.shape[0]}")
    print(f"    Feature dim:   {X_train.shape[1]}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 2. Standardize
    print("\n[2] Standardizing features...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"    Time: {time.time() - t0:.2f}s")

    # 3. Train
    print("\n[3] Training Logistic Regression...")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"    Time: {time.time() - t0:.2f}s")

    # 4. Test
    print("\n[4] Testing model...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Test accuracy: {acc * 100:.2f}%")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 5. Save model (weights + bias + scaler)
    print("\n[5] Saving parameters...")
    model_data = {
        "model": model,
        "scaler": scaler,
        "accuracy": acc
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to: {model_path}")
    print(f"    File size: {os.path.getsize(model_path) / 1024:.1f} KB")

    # 6. Save result
    result_path = "../results/accuracy.txt"
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"Logistic Regression MNIST Test Accuracy: {acc * 100:.2f}%\n")
    print(f"\n[6] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
