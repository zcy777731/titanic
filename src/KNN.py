import os
import pickle
import time
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


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
    print("KNN CIFAR-10 Classification")
    print("=" * 50)

    data_dir = "../data/cifar10_images"
    model_dir = "../models"
    result_path = "../results/accuracy.txt"
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load data
    print("\n[1] Loading CIFAR-10 data...")
    t0 = time.time()
    X_train, y_train, X_test, y_test, classes = load_cifar10(data_dir)
    print(f"    Train samples: {X_train.shape[0]}")
    print(f"    Test samples:  {X_test.shape[0]}")
    print(f"    Classes: {classes}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 2. Standardize
    print("\n[2] Standardizing features...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"    Time: {time.time() - t0:.2f}s")

    # 3. Train KNN (use subset for speed)
    print("\n[3] Training KNN model...")
    t0 = time.time()

    sample_per_class = 1000
    X_train_sample, y_train_sample = [], []
    for i in range(10):
        mask = y_train == i
        X_train_sample.append(X_train[mask][:sample_per_class])
        y_train_sample.append(y_train[mask][:sample_per_class])
    X_train_sample = np.vstack(X_train_sample)
    y_train_sample = np.concatenate(y_train_sample)

    print(f"    Actual training samples: {X_train_sample.shape[0]}")

    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_train_sample, y_train_sample)
    print(f"    Time: {time.time() - t0:.2f}s")

    # 4. Test
    print("\n[4] Testing model...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Test accuracy: {acc * 100:.2f}%")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 5. Save model
    model_data = {
        "model": model,
        "scaler": scaler,
        "classes": classes,
        "accuracy": acc
    }
    model_path = os.path.join(model_dir, "knn_cifar10.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n[5] Model saved to: {model_path}")

    # 6. Save result
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"KNN CIFAR-10 Test Accuracy: {acc * 100:.2f}%\n")
    print(f"[6] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
