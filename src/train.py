import os
import pickle
import pandas as pd
from sklearn.svm import SVC
from data_preprocess import preprocess_titanic

print("=" * 50)
print("SVM Titanic Train")
print("=" * 50)

# 路径
train_path = "../data/titanic/titanic_train.csv"
test_path = "../data/titanic/titanic_test.csv"
model_path = "../models/svm_titanic.pkl"

# 读取数据
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 预处理
X_train, y_train, X_test, y_test, mean, std = preprocess_titanic(train_df, test_df)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"特征维度: {X_train.shape[1]}")

# 建立SVM模型
model = SVC(kernel="rbf", C=1.0, gamma="scale")

# 训练
print("\n开始训练SVM模型...")
model.fit(X_train, y_train)
print("训练完成！")

# 保存模型和标准化参数
os.makedirs("../models", exist_ok=True)
save_data = {
    "model": model,
    "mean": mean,
    "std": std
}

with open(model_path, "wb") as f:
    pickle.dump(save_data, f)

print(f"模型已保存到: {model_path}")
