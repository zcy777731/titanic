"""
ANN CIFAR-10 Testing
Usage: python ann_cifar10_test.py
"""

import os
import pickle
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_cifar10(data_dir):
    X_test, y_test = [], []
    classes = sorted(os.listdir(os.path.join(data_dir, "test")))

    for label, cls in enumerate(classes):
        test_cls_dir = os.path.join(data_dir, "test", cls)
        for fname in os.listdir(test_cls_dir):
            img = Image.open(os.path.join(test_cls_dir, fname)).convert("RGB")
            X_test.append(np.array(img).flatten())
            y_test.append(label)

    return np.array(X_test, dtype=np.float32), np.array(y_test), classes


def main():
    print("=" * 50)
    print("ANN CIFAR-10 Test")
    print("=" * 50)

    data_dir = "../data/cifar10_images"
    model_path = "../models/ann_cifar10.pkl"
    result_dir = "../results"
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load model
    print("\n[1] Loading model...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    scaler = model_data["scaler"]
    pca = model_data["pca"]
    classes = model_data["classes"]
    print(f"    Model loaded (train accuracy: {model_data.get('accuracy', 'N/A')})")

    # 2. Load test data
    print("\n[2] Loading test data...")
    t0 = time.time()
    X_test, y_test, _ = load_cifar10(data_dir)
    print(f"    Test samples: {X_test.shape[0]}")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 3. Preprocess
    print("\n[3] Preprocessing...")
    t0 = time.time()
    X_test = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test)
    print(f"    Time: {time.time() - t0:.2f}s")

    # 4. Predict
    print("\n[4] Predicting...")
    t0 = time.time()
    y_pred = model.predict(X_test_pca)
    acc = np.mean(y_pred == y_test)
    print(f"    Test accuracy: {acc * 100:.2f}%")
    print(f"    Time: {time.time() - t0:.2f}s")

    # 5. Plot loss curve from saved model
    print("\n[5] Plotting model loss curve...")
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Training Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('ANN CIFAR-10 - Loss Curve (Test)')
        plt.legend()
        plt.grid(alpha=0.3)
        loss_path = os.path.join(result_dir, "ann_cifar10_loss_test.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Loss curve saved to: {loss_path}")
    else:
        print("    No loss curve available.")

    # 6. Save result
    result_path = os.path.join(result_dir, "accuracy.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"ANN CIFAR-10 Test (loaded model): {acc * 100:.2f}%\n")
    print(f"\n[6] Result saved to: {result_path}")


if __name__ == "__main__":
    main()
