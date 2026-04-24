import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from data_preprocess import preprocess_titanic

print("=" * 50)
print("SVM Titanic Test")
print("=" * 50)

train_path = "../data/titanic/titanic_train.csv"
test_path = "../data/titanic/titanic_test.csv"
model_path = "../models/svm_titanic.pkl"
result_path = "../results/accuracy.txt"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train, y_train, X_test, y_test, mean, std, = preprocess_titanic(train_df, test_df)

with open(model_path, "rb") as f:
    save_data = pickle.load(f)

model = save_data["model"]

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {acc * 100:.2f}%")

os.makedirs("../results", exist_ok=True)
with open(result_path, "a", encoding="utf-8") as f:
    f.write(f"SVM Titanic Test Accuracy: {acc * 100:.2f}%\n")

print(f"Result saved: {result_path}")
