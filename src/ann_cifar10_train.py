"""
ANN CIFAR-10 Training
Usage: python ann_cifar10_train.py
"""

import os
import pickle
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_cifar10(data_dir):
    X_train, y_train = [], []
    X_test, y_test = [], []
    classes = sorted(os.listdir(os.path.join(data_dir, "train")))

    for label, cls in enumerate(classes):
        train_cls_dir = os.path.join(data_dir, "train", cls)
        for fname in os.listdir(train_cls_dir):
            img = Image.open(os.path.join(train_cls_dir, fname)).convert("RGB")
            X_train.append(np.array(img).flatten())
            y_train.append(label)

        test_cls_dir = os.path.join(data_dir, "test", cls)
        for fname in os.listdir(test_cls_dir):
            img = Image.open(os.path.join(test_cls_dir, fname)).convert("RGB")
            X_test.append(np.array(img).flatten())
            y_test.append(label)

    return (np.array(X_train, dtype=np.float32), np.array(y_train),
            np.array(X_test, dtype=np.float32), np.array(y_test),
            classes)


def main():
    print("=" * 50)
    print("ANN CIFAR-10 Train")
    print("=" * 50)

    data_dir = "../data/cifar10_images"
    model_dir = "../models"
    result_dir = "../results"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load data
    print("\n[1] Loading CIFAR-10 data...")
    t0 = time.time()
    X_train, y_train, X_test, y_test, classes = load_cifar10(data_dir)
    print(f"    Train samples: {X_train.shape[0]}")
    print(f"    Test samples:  {X_test.shape[0]}")
    print(f"    Feature dim:   {X_train.shape[1]}")
    print(f"    Classes: {classes}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 2. Standardize
    print("\n[2] Standardizing features...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"    Time: {time.time() - t0:.2f}s")

    # 3. PCA
    print("\n[3] PCA dimensionality reduction...")
    t0 = time.time()
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"    Reduced dimension: {X_train_pca.shape[1]}")
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 4. Train ANN (use subset for speed)
    print("\n[4] Training ANN (MLPClassifier)...")
    t0 = time.time()

    sample_per_class = 500
    X_train_sample, y_train_sample = [], []
    for i in range(10):
        mask = y_train == i
        X_train_sample.append(X_train_pca[mask][:sample_per_class])
        y_train_sample.append(y_train[mask][:sample_per_class])
    X_train_sample = np.vstack(X_train_sample)
    y_train_sample = np.concatenate(y_train_sample)
    print(f"    Actual training samples: {X_train_sample.shape[0]}")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False
    )
    model.fit(X_train_sample, y_train_sample)
    print(f"    Iterations: {model.n_iter_}")
    print(f"    Final loss: {model.loss_:.4f}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 5. Evaluate
    print("\n[5] Evaluating...")
    t0 = time.time()
    y_pred = model.predict(X_test_pca)
    acc = np.mean(y_pred == y_test)
    print(f"    Test accuracy: {acc * 100:.2f}%")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 6. Plot loss curve
    print("\n[6] Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ANN CIFAR-10 - Loss Curve (Training)')
    plt.legend()
    plt.grid(alpha=0.3)
    loss_path = os.path.join(result_dir, "ann_cifar10_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Loss curve saved to: {loss_path}")

    # 7. Save model + preprocessing params
    print("\n[7] Saving model...")
    model_data = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "classes": classes,
        "accuracy": acc
    }
    model_path = os.path.join(model_dir, "ann_cifar10.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to: {model_path}")

    # 8. Save result
    result_path = os.path.join(result_dir, "accuracy.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"ANN CIFAR-10 Test Accuracy: {acc * 100:.2f}%\n")
    print(f"\n[8] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
