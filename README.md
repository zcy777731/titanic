# Machine Learning Project

Machine learning classification and regression project using four datasets: **Titanic**, **House Prices**, **MNIST**, and **CIFAR-10**.

## Project Structure

```
ML_Project/
├── main.py                  # Entry point - menu to run all models
├── requirements.txt         # Python dependencies
├── README.md
├── data/
│   ├── titanic/             # Titanic survival dataset (CSV)
│   ├── house/               # House price dataset (CSV)
│   ├── mnist_images/        # MNIST handwritten digits (PNG, 28x28)
│   └── cifar10_images/      # CIFAR-10 color images (PNG, 32x32)
├── src/
│   ├── data.preprocess.py   # Titanic data preprocessing
│   ├── train.py             # Titanic SVM training
│   ├── test.py              # Titanic SVM testing
│   ├── logistic_regression.py  # Titanic Logistic Regression
│   ├── linear_regression.py    # House Price Linear Regression
│   ├── logistic_mnist.py       # MNIST Logistic Regression
│   ├── svm_mnist.py            # MNIST SVM
│   ├── KNN.py                  # CIFAR-10 KNN
│   └── SVM.py                  # CIFAR-10 SVM
├── models/                  # Saved trained models (.pkl)
└── results/                 # Accuracy results and figures
```

## Datasets

| Dataset | Description | Features | Classes |
|---------|-------------|----------|---------|
| Titanic | Passenger survival prediction | Age, Sex, Fare, Embarked, etc. | 2 (Survived/Not) |
| House | House price prediction | x1, x2, x3, x4 | Regression |
| MNIST | Handwritten digit recognition | 28x28 grayscale pixels (784) | 10 (0-9) |
| CIFAR-10 | Object recognition | 32x32 RGB pixels (3072) | 10 classes |

## Models Implemented

| Model | Dataset | Algorithm |
|-------|---------|-----------|
| SVM | Titanic | SVC (RBF kernel) |
| Logistic Regression | Titanic | sklearn + manual (mini-batch GD) |
| Linear Regression | House | Least squares / BGD / SGD / Mini-batch |
| Logistic Regression | MNIST | Softmax (multinomial) |
| SVM | MNIST | LinearSVC + PCA |
| KNN | CIFAR-10 | KNeighborsClassifier (k=5) |
| SVM | CIFAR-10 | LinearSVC + PCA |

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run via main menu

```bash
python main.py
```

Select a model number from the menu to train and evaluate.

### Run individual scripts

```bash
# Titanic
python src/train.py
python src/test.py
python src/logistic_regression.py

# House Price
python src/linear_regression.py

# MNIST
python src/logistic_mnist.py
python src/svm_mnist.py

# CIFAR-10
python src/KNN.py
python src/SVM.py
```

## Results

| Model | Dataset | Test Accuracy |
|-------|---------|---------------|
| Logistic Regression | MNIST | 91.98% |
| SVM | MNIST | 91.23% |
| SVM | CIFAR-10 | 37.12% |
| KNN | CIFAR-10 | 29.10% |
