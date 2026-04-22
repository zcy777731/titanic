import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from data_preprocess import preprocess_titanic

print("=" * 50)
print("SVM Titanic Test")
print("=" * 50)

# 路径
train_path = "../data/titanic/titanic_train.csv"
test_path = "../data/titanic/titanic_test.csv"
model_path = "../models/svm_titanic.pkl"
result_path = "../results/accuracy.txt"

# 读取数据
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 预处理
X_train, y_train, X_test, y_test, mean, std, = preprocess_titanic(train_df, test_df)

# 加载模型
with open(model_path, "rb") as f:
    save_data = pickle.load(f)

model = save_data["model"]

# 预测
y_pred = model.predict(X_test)

# 准确率
acc = accuracy_score(y_test, y_pred)

print(f"测试集准确率: {acc * 100:.2f}%")

# 保存结果
os.makedirs("../results", exist_ok=True)
with open(result_path, "w", encoding="utf-8") as f:
    f.write(f"SVM Titanic Test Accuracy: {acc * 100:.2f}%\n")

print(f"结果已保存到: {result_path}")
