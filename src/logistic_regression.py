import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. Load data
# =========================
train_df = pd.read_csv('../data/titanic/titanic_train.csv')
test_df = pd.read_csv('../data/titanic/titanic_test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =========================
# 2. Drop zero-prefix columns
# =========================
zero_cols = [col for col in train_df.columns if col.startswith('zero')]

train_df = train_df.drop(columns=zero_cols)
test_df = test_df.drop(columns=zero_cols, errors='ignore')

print("\nAfter dropping zero columns:")
print("Train shape:", train_df.shape)
print("Features:", train_df.columns.tolist())

# =========================
# 3. Data cleaning
# =========================
age_mean = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(age_mean)
test_df['Age'] = test_df['Age'].fillna(age_mean)

embarked_mode = train_df['Embarked'].mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)

fare_median = train_df[(train_df['Fare'] > 0) & (train_df['Fare'] < 500)]['Fare'].median()
train_df.loc[(train_df['Fare'] == 0) | (train_df['Fare'] > 500), 'Fare'] = fare_median
test_df.loc[(test_df['Fare'] == 0) | (test_df['Fare'] > 500), 'Fare'] = fare_median

print("\nData cleaning done")

# =========================
# 4. One-Hot Encoding
# =========================
sex_dummies_train = pd.get_dummies(train_df['Sex'], prefix='Sex')
sex_dummies_test = pd.get_dummies(test_df['Sex'], prefix='Sex')

pclass_dummies_train = pd.get_dummies(train_df['Pclass'], prefix='Pclass')
pclass_dummies_test = pd.get_dummies(test_df['Pclass'], prefix='Pclass')

embarked_dummies_train = pd.get_dummies(train_df['Embarked'], prefix='Embarked')
embarked_dummies_test = pd.get_dummies(test_df['Embarked'], prefix='Embarked')

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

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

print("\nFeature dimensions:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =========================
# 5. Labels
# =========================
y_train = train_df['Survived'].values
y_test = test_df['Survived'].values

print("\nLabel distribution:")
print("Train:", np.bincount(y_train))
print("Test:", np.bincount(y_test))

# =========================
# 6. Normalization
# =========================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nNormalization done")

# =========================
# 7. sklearn Logistic Regression
# =========================
print("\n" + "=" * 50)
print("sklearn Logistic Regression")
print("=" * 50)

model_sklearn = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model_sklearn.fit(X_train_scaled, y_train)

y_pred_train = model_sklearn.predict(X_train_scaled)
accuracy_train = accuracy_score(y_train, y_pred_train)

y_pred_sklearn = model_sklearn.predict(X_test_scaled)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("Train accuracy:", accuracy_train)
print("Test accuracy:", accuracy_sklearn)

# =========================
# 8. Manual Logistic Regression (mini-batch GD + L2)
# =========================
print("\n" + "=" * 50)
print("Manual Logistic Regression (mini-batch + L2)")
print("=" * 50)


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


model_manual = ManualLogisticRegression(
    learning_rate=0.5,
    n_epochs=500,
    batch_size=64,
    lambda_reg=0.1
)
model_manual.fit(X_train_scaled, y_train)

y_pred_manual = model_manual.predict(X_test_scaled)
accuracy_manual = accuracy_score(y_test, y_pred_manual)

print(f"\nManual logistic regression test accuracy: {accuracy_manual}")

# =========================
# 9. Loss curve
# =========================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(model_manual.loss_history) + 1), model_manual.loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss vs Epoch (Mini-batch Gradient Descent with L2 Regularization)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/logistic_mnist_loss.png', dpi=150)
plt.close()

print("\nLoss curve saved to results/logistic_mnist_loss.png")

# =========================
# 10. Summary
# =========================
print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"sklearn logistic regression test accuracy: {accuracy_sklearn:.4f}")
print(f"Manual logistic regression test accuracy: {accuracy_manual:.4f}")
