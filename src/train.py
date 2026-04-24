import os
import pickle
import pandas as pd
from sklearn.svm import SVC
from data_preprocess import preprocess_titanic

print("=" * 50)
print("SVM Titanic Train")
print("=" * 50)

train_path = "../data/titanic/titanic_train.csv"
test_path = "../data/titanic/titanic_test.csv"
model_path = "../models/svm_titanic.pkl"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train, y_train, X_test, y_test, mean, std = preprocess_titanic(train_df, test_df)

print(f"Train samples: {X_train.shape[0]}")
print(f"Feature dim: {X_train.shape[1]}")

model = SVC(kernel="rbf", C=1.0, gamma="scale")

print("\nTraining SVM model...")
model.fit(X_train, y_train)
print("Training done!")

os.makedirs("../models", exist_ok=True)
save_data = {
    "model": model,
    "mean": mean,
    "std": std
}

with open(model_path, "wb") as f:
    pickle.dump(save_data, f)

print(f"Model saved: {model_path}")
