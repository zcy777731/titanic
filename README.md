<p align="center">
  <h1 align="center">🤖 Machine Learning Project</h1>
  <p align="center">
    A comprehensive ML project covering <strong>Classification</strong>, <strong>Regression</strong>,
    and <strong>Deep Learning</strong> across 4 datasets.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python">
  <img src="https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikitlearn">
  <img src="https://img.shields.io/badge/MNIST-91.98%25-brightgreen">
  <img src="https://img.shields.io/badge/CIFAR--10-37.37%25-yellow">
</p>

---

## 📋 Table of Contents

- [Datasets](#-datasets)
- [Models & Results](#-models--results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Loss Curves](#-loss-curves)

---

## 📊 Datasets

| Dataset | Description | Type | Samples | Features | Classes |
|---------|-------------|------|---------|----------|---------|
| [Titanic](https://www.kaggle.com/c/titanic) | Passenger survival prediction | Classification | 891 | Age, Sex, Fare, Embarked... | 2 |
| [House Prices](data/house/) | House price regression | Regression | 1000 | x1, x2, x3, x4 | — |
| [MNIST](http://yann.lecun.com/exdb/mnist/) | Handwritten digit recognition | Classification | 60,000 | 28×28 grayscale pixels (784) | 10 |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | Object recognition | Classification | 60,000 | 32×32 RGB pixels (3072) | 10 |
| [Dry Bean](https://www.muratkoklu.com/datasets/) | Dry bean classification | Classification | 13,611 | 16 shape features (Area, Perimeter, etc.) | 7 |

---

## 🧠 Models & Results

### Classical ML Models

| Model | Dataset | Algorithm | Accuracy |
|-------|---------|-----------|:--------:|
| SVM | Titanic | SVC (RBF kernel) | **83.80%** |
| Logistic Regression | Titanic | sklearn + Manual (mini-batch GD) | **79.89%** |
| Linear Regression | House | Least Squares / BGD / SGD / Mini-batch | RMSE **0.495** |
| Logistic Regression | MNIST | Softmax Multinomial | **91.98%** |
| SVM | MNIST | LinearSVC + PCA | **91.23%** |
| KNN | CIFAR-10 | KNeighbors (k=5) | **29.10%** |
| SVM | CIFAR-10 | LinearSVC + PCA | **37.12%** |
| Logistic Regression | Dry Bean | L2 Multinomial | **91.85%** |
| SVM (RBF) | Dry Bean | RBF kernel | **92.88%** |
| KNN | Dry Bean | k=5, kd_tree | **91.74%** |
| XGBoost ⭐ | Dry Bean | n_estimators=200 | **92.58%** |

### ANN Models (This Assignment)

| Model | Dataset | Architecture | Accuracy / Metric |
|-------|---------|:------------:|:-----------------:|
| ANN | Titanic | MLP (64, 32) | **83.24%** |
| ANN | House | MLPRegressor (64, 32) | MSE **0.255** |
| ANN | CIFAR-10 | MLP (128, 64) + PCA | **37.37%** |

---

## 📁 Project Structure

```
ML_Project/
├── main.py                     # CLI entry point
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── titanic/                # Titanic CSV data
│   ├── house/                  # House price CSV data
│   ├── mnist_images/           # MNIST handwritten digits
│   └── cifar10_images/         # CIFAR-10 object images
├── DryBeanDataset/              # Dry Bean dataset
│   ├── Dry_Bean_Dataset_Dirty_train.csv  # Raw train
│   ├── Dry_Bean_Dataset_Dirty_val.csv    # Raw val
│   ├── Dry_Bean_Dataset_Dirty_test.csv   # Raw test
│   ├── train_clean.csv                   # Cleaned train
│   ├── val_clean.csv                     # Cleaned val
│   └── test_clean.csv                    # Cleaned test
├── src/
│   ├── classical/
│   │   ├── train.py / test.py          # Titanic SVM
│   │   ├── logistic_regression.py      # Titanic Logistic Regression
│   │   ├── linear_regression.py        # House Price Linear Regression
│   │   ├── logistic_mnist.py           # MNIST Logistic Regression
│   │   ├── svm_mnist.py                # MNIST SVM
│   │   ├── KNN.py / SVM.py             # CIFAR-10
│   │   └── data_preprocess.py          # Titanic preprocessing
│   └── ann/
│       ├── ann_house.py                # ANN House Regression
│       ├── ann_titanic_train.py         # ANN Titanic Training
│       ├── ann_titanic_test.py          # ANN Titanic Testing
│       ├── ann_cifar10_train.py         # ANN CIFAR-10 Training
│       └── ann_cifar10_test.py          # ANN CIFAR-10 Testing
│   ├── drybean_analysis.py              # DryBean: Data analysis
│   ├── drybean_preprocessing.py         # DryBean: Data cleaning
│   └── drybean_experiments.py           # DryBean: Experiments
├── models/                     # Saved trained models (.pkl)
└── results/                    # Accuracy logs & loss curve plots
```

> **Note:** Large image datasets (`mnist_images/`, `cifar10_images/`) and model files (`*.pkl`) are excluded from git via `.gitignore`.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/zcy777731/titanic.git
cd titanic

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a model
python main.py --algo=ann --data=titanic --process=train
```

---

## 🎮 Usage Examples

### ANN Models

```bash
# House Price Regression
python main.py --algo=ann --data=house --process=train

# Titanic (train then test)
python main.py --algo=ann --data=titanic --process=train
python main.py --algo=ann --data=titanic --process=test

# CIFAR-10 (train then test)
python main.py --algo=ann --data=cifar10 --process=train
python main.py --algo=ann --data=cifar10 --process=test
```

### Classical Models

```bash
# SVM Titanic
python main.py --algo=svm --data=titanic --process=train

# Logistic Regression MNIST
python main.py --algo=logistic --data=mnist --process=train

# KNN / SVM CIFAR-10
python main.py --algo=knn --data=cifar10 --process=train
python main.py --algo=svm --data=cifar10 --process=train
```

### Dry Bean Classification

```bash
# Data analysis
python main.py --algo=drybean --data=drybean --process=analyze
# Data preprocessing
python main.py --algo=drybean --data=drybean --process=preprocess
# Run experiments (LR + SVM + KNN + XGBoost)
python main.py --algo=drybean --data=drybean --process=experiments
# Train single model
python main.py --algo=lr --data=drybean --process=train
python main.py --algo=xgb --data=drybean --process=train
```

### Run Everything

```bash
python main.py --algo=all --data=all --process=all
```

---

## 📈 Loss Curves

| Model | Loss Curve |
|-------|:----------:|
| ANN Titanic | ![Titanic Loss](results/ann_titanic_loss.png) |
| ANN House | ![House Loss](results/ann_house_loss.png) |
| ANN CIFAR-10 | ![CIFAR-10 Loss](results/ann_cifar10_loss.png) |

---

## 🛠️ Tech Stack

- **Python 3.12** — Core language
- **scikit-learn 1.8** — ML models (SVM, KNN, Logistic Regression, MLP, PCA)
- **Pandas** — Data preprocessing
- **NumPy** — Numerical computing
- **Matplotlib** — Visualization & loss curves
- **Pillow** — Image loading

---

<p align="center">
  <i>Course Project — ML Models Implementation & Comparison</i>
</p>
