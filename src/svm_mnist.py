import os
import pickle
import time
import sys
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
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
    print("SVM MNIST Classification")
    print("=" * 50)

    data_dir = "../data/mnist_images"
    model_path = "../models/svm_mnist.pkl"
    os.makedirs("../models", exist_ok=True)

    # 1. Load data
    print("\n[1] Loading MNIST data...")
    sys.stdout.flush()
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    print(f"    Train samples: {X_train.shape[0]}")
    print(f"    Test samples:  {X_test.shape[0]}")
    print(f"    Feature dim:   {X_train.shape[1]}")
    print(f"    Time: {time.time() - t0:.2f}s")
    sys.stdout.flush()

    # 2. Standardize
    print("\n[2] Standardizing features...")
    sys.stdout.flush()
    t0 = time.time()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"    Time: {time.time() - t0:.2f}s")
    sys.stdout.flush()

    # 3. PCA dimensionality reduction
    print("\n[3] PCA dimensionality reduction...")
    sys.stdout.flush()
    t0 = time.time()
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"    Reduced dimension: {X_train_pca.shape[1]}")
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"    Time: {time.time() - t0:.2f}s")
    sys.stdout.flush()

    # 4. Train SVM (use subset for speed)
    print("\n[4] Training SVM (LinearSVC)...")
    sys.stdout.flush()
    t0 = time.time()

    sample_per_class = 1000
    X_train_sample, y_train_sample = [], []
    for i in range(10):
        mask = y_train == i
        X_train_sample.append(X_train_pca[mask][:sample_per_class])
        y_train_sample.append(y_train[mask][:sample_per_class])
    X_train_sample = np.vstack(X_train_sample)
    y_train_sample = np.concatenate(y_train_sample)

    print(f"    Actual training samples: {X_train_sample.shape[0]}")
    sys.stdout.flush()

    model = LinearSVC(C=1.0, max_iter=5000, random_state=42, dual="auto")
    model.fit(X_train_sample, y_train_sample)
    print(f"    Time: {time.time() - t0:.2f}s")
    sys.stdout.flush()

    # 5. Test
    print("\n[5] Testing model...")
    sys.stdout.flush()
    t0 = time.time()
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Test accuracy: {acc * 100:.2f}%")
    print(f"    Time: {time.time() - t0:.2f}s")
    sys.stdout.flush()

    # 6. Save model
    print("\n[6] Saving parameters...")
    sys.stdout.flush()
    model_data = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "accuracy": acc
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to: {model_path}")
    print(f"    File size: {os.path.getsize(model_path) / 1024:.1f} KB")
    sys.stdout.flush()

    # 7. Save result
    result_path = "../results/accuracy.txt"
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"SVM MNIST Test Accuracy: {acc * 100:.2f}%\n")
    print(f"\n[7] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
