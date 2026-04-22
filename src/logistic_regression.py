import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import subprocess
import platform
warnings.filterwarnings('ignore')

# =========================
# 1. 读取数据
# =========================
train_df = pd.read_csv('titanic_train.csv')
test_df = pd.read_csv('titanic_test.csv')

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

# =========================
# 2. 删除zero开头的列
# =========================
zero_cols = [col for col in train_df.columns if col.startswith('zero')]

train_df = train_df.drop(columns=zero_cols)
test_df = test_df.drop(columns=zero_cols, errors='ignore')

print("\n删除zero列后:")
print("训练集形状:", train_df.shape)
print("有效特征列:", train_df.columns.tolist())

# =========================
# 3. 数据清洗
# =========================
# 3.1 处理Age缺失值（用平均值代替）
age_mean = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(age_mean)
test_df['Age'] = test_df['Age'].fillna(age_mean)

# 3.2 处理Embarked缺失值（用众数代替）
embarked_mode = train_df['Embarked'].mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)

# 3.3 处理Fare异常值（Fare为0或异常高，用中位数代替）
fare_median = train_df[(train_df['Fare'] > 0) & (train_df['Fare'] < 500)]['Fare'].median()
train_df.loc[(train_df['Fare'] == 0) | (train_df['Fare'] > 500), 'Fare'] = fare_median
test_df.loc[(test_df['Fare'] == 0) | (test_df['Fare'] > 500), 'Fare'] = fare_median

print("\n数据清洗完成")

# =========================
# 4. 特征工程：One-Hot编码
# =========================
# 保留的有效特征：Age, Fare, Sex, sibsp, Parch, Pclass, Embarked (共7个)

# 对Sex进行One-Hot编码
sex_dummies_train = pd.get_dummies(train_df['Sex'], prefix='Sex')
sex_dummies_test = pd.get_dummies(test_df['Sex'], prefix='Sex')

# 对Pclass进行One-Hot编码
pclass_dummies_train = pd.get_dummies(train_df['Pclass'], prefix='Pclass')
pclass_dummies_test = pd.get_dummies(test_df['Pclass'], prefix='Pclass')

# 对Embarked进行One-Hot编码
embarked_dummies_train = pd.get_dummies(train_df['Embarked'], prefix='Embarked')
embarked_dummies_test = pd.get_dummies(test_df['Embarked'], prefix='Embarked')

# 构建最终特征矩阵
numeric_cols = ['Age', 'Fare', 'sibsp', 'Parch']

X_train_numeric = train_df[numeric_cols].values
X_test_numeric = test_df[numeric_cols].values

X_train = np.hstack([
    X_train_numeric,
    sex_dummies_train.values,
    pclass_dummies_train.values,
    embarked_dummies_train.values
])

X_test = np.hstack([
    X_test_numeric,
    sex_dummies_test.values,
    pclass_dummies_test.values,
    embarked_dummies_test.values
])

print("\n特征维度:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =========================
# 5. 划分标签
# =========================
y_train = train_df['2urvived'].values
y_test = test_df['2urvived'].values

print("\n标签分布:")
print("训练集:", np.bincount(y_train))
print("测试集:", np.bincount(y_test))

# =========================
# 6. 归一化
# =========================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n归一化完成")

# =========================
# 7. sklearn逻辑回归训练和测试
# =========================
print("\n" + "="*50)
print("7. sklearn 逻辑回归")
print("="*50)

model_sklearn = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model_sklearn.fit(X_train_scaled, y_train)

y_pred_train = model_sklearn.predict(X_train_scaled)
accuracy_train = accuracy_score(y_train, y_pred_train)

y_pred_sklearn = model_sklearn.predict(X_test_scaled)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("训练集准确率:", accuracy_train)
print("测试集准确率:", accuracy_sklearn)

# =========================
# 8. 手写mini-batch梯度下降（带L2正则化）
# =========================
print("\n" + "="*50)
print("8. 手写逻辑回归 (mini-batch + L2正则)")
print("="*50)

class ManualLogisticRegression:
    def __init__(self, learning_rate=0.1, n_epochs=500, batch_size=32, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)

        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        loss = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        loss += self.lambda_reg / (2 * m) * np.sum(self.weights ** 2)

        return loss

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                batch_m = X_batch.shape[0]

                z = np.dot(X_batch, self.weights) + self.bias
                h = self.sigmoid(z)

                dw = 1/batch_m * np.dot(X_batch.T, (h - y_batch)) + self.lambda_reg / batch_m * self.weights
                db = 1/batch_m * np.sum(h - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.4f}")

        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        return (h >= 0.5).astype(int)

# 训练手写模型
model_manual = ManualLogisticRegression(
    learning_rate=0.5,
    n_epochs=500,
    batch_size=64,
    lambda_reg=0.1
)
model_manual.fit(X_train_scaled, y_train)

y_pred_manual = model_manual.predict(X_test_scaled)
accuracy_manual = accuracy_score(y_test, y_pred_manual)

print(f"\n手写逻辑回归测试集准确率: {accuracy_manual}")

# =========================
# 9. 绘制Loss曲线
# =========================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(model_manual.loss_history) + 1), model_manual.loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss vs Epoch (Mini-batch Gradient Descent with L2 Regularization)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()

# 自动打开图片
if platform.system() == 'Darwin':
    subprocess.run(['open', 'loss_curve.png'])
elif platform.system() == 'Windows':
    subprocess.run(['start', 'loss_curve.png'], shell=True)
else:
    subprocess.run(['xdg-open', 'loss_curve.png'])

print("\nLoss曲线已保存到 loss_curve.png")

# =========================
# 10. 结果汇总
# =========================
print("\n" + "="*50)
print("结果汇总")
print("="*50)
print(f"sklearn逻辑回归测试准确率: {accuracy_sklearn:.4f}")
print(f"手写逻辑回归测试准确率: {accuracy_manual:.4f}")