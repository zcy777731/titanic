"""
期末作业 Part 3: 多算法实验对比
4种算法: LR / SVM / KNN / XGBoost（课上没讲过）
对比: 准确率 / 速度 / 鲁棒性 / 过拟合
"""
import sys, time, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("Dry Bean — 多算法实验对比")
print("="*60)

train = pd.read_csv('DryBeanDataset/train_clean.csv')
val = pd.read_csv('DryBeanDataset/val_clean.csv')
test = pd.read_csv('DryBeanDataset/test_clean.csv')

feat_cols = [c for c in train.columns if c != 'Class']
X_train, y_train_str = train[feat_cols].values, train['Class'].values
X_val, y_val_str = val[feat_cols].values, val['Class'].values
X_test, y_test_str = test[feat_cols].values, test['Class'].values

# 标签编码
le = LabelEncoder()
y_train = le.fit_transform(y_train_str)
y_val = le.transform(y_val_str)
y_test = le.transform(y_test_str)
classes = le.classes_
print(f"标签: {dict(zip(classes, range(len(classes))))}")
print(f"训练: {len(X_train)}  验证: {len(X_val)}  测试: {len(X_test)}  特征: {len(feat_cols)}")

# 模型定义
models = {}
results = []

for name in ['Logistic Regression', 'SVM (RBF)', 'KNN (k=5)', 'XGBoost']:
    print(f"\n{'='*40}")
    print(f"训练: {name}")
    print('='*40)

    if name == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                   penalty='l2', random_state=42)
    elif name == 'SVM (RBF)':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    elif name == 'KNN (k=5)':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    elif name == 'XGBoost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, verbosity=0)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_t = time.time() - t0

    t0 = time.time()
    yp_tr = model.predict(X_train)
    yp_va = model.predict(X_val)
    yp_te = model.predict(X_test)
    infer_t = time.time() - t0

    from sklearn.metrics import accuracy_score
    a_tr = accuracy_score(y_train, yp_tr)
    a_va = accuracy_score(y_val, yp_va)
    a_te = accuracy_score(y_test, yp_te)

    models[name] = model
    results.append([name, a_tr, a_va, a_te, train_t, infer_t])
    print(f"  train={a_tr*100:.2f}%  val={a_va*100:.1f}%  test={a_te*100:.2f}%  train_t={train_t:.2f}s  infer_t={infer_t*1000:.1f}ms")
    print(f"  过拟合: {(a_tr-a_te)*100:.2f}%")

# 汇总
print(f"\n{'='*70}")
print(f"{'算法':25s} {'train%':8s} {'val%':8s} {'test%':8s} {'train_s':8s} {'infer_ms':8s}")
print('-'*70)
for r in results:
    print(f"{r[0]:25s} {r[1]*100:7.2f}% {r[2]*100:7.2f}% {r[3]*100:7.2f}% {r[4]:7.2f}s {r[5]*1000:7.1f}ms")

# 准确率图
plt.figure(figsize=(10, 5))
x = np.arange(len(results))
plt.bar(x-0.25, [r[1]*100 for r in results], 0.2, label='Train')
plt.bar(x, [r[2]*100 for r in results], 0.2, label='Val')
plt.bar(x+0.25, [r[3]*100 for r in results], 0.2, label='Test')
plt.xticks(x, [r[0] for r in results], rotation=15)
plt.ylabel('Accuracy (%)'); plt.title('Algorithm Comparison')
plt.legend(); plt.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('tmp_imgs/drybean_accuracy.png', dpi=100)

# 鲁棒性
print(f"\n--- 鲁棒性测试 ---")
noises = [0.0, 0.1, 0.2, 0.5, 1.0]
rob = {n: [] for n in models}
for noise in noises:
    Xn = X_test + np.random.randn(*X_test.shape) * noise
    for name, model in models.items():
        rob[name].append(accuracy_score(y_test, model.predict(Xn)))

plt.figure(figsize=(10, 5))
for name in models:
    plt.plot(noises, [a*100 for a in rob[name]], 'o-', label=name)
plt.xlabel('Noise (σ)'); plt.ylabel('Test Accuracy (%)')
plt.title('Robustness: Accuracy vs Noise'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('tmp_imgs/drybean_robustness.png', dpi=100)

for noise in noises:
    print(f"  σ={noise:.1f}: ", end='')
    for name in models:
        print(f"{name}={rob[name][noises.index(noise)]*100:.1f}% ", end='')
    print()

# 速度图
plt.figure(figsize=(10, 5))
ts = [r[5]*1000 for r in results]
plt.barh([r[0] for r in results], ts, color=plt.cm.Set2(np.linspace(0, 1, len(ts))))
plt.xlabel('Inference (ms)'); plt.title('Speed Comparison')
for i, v in enumerate(ts):
    plt.text(v+2, i, f'{v:.1f}ms', va='center')
plt.tight_layout(); plt.savefig('tmp_imgs/drybean_speed.png', dpi=100)

# 过拟合图
plt.figure(figsize=(10, 5))
plt.plot(x, [r[1]*100 for r in results], 's-', label='Train', ms=8)
plt.plot(x, [r[3]*100 for r in results], 'o-', label='Test', ms=8)
for i, r in enumerate(results):
    d = (r[1]-r[3])*100
    plt.plot([i, i], [r[1]*100, r[3]*100], 'r--', alpha=0.3)
    plt.text(i, (r[1]*100+r[3]*100)/2+0.5, f'{d:.1f}%', ha='center')
plt.xticks(x, [r[0] for r in results], rotation=15)
plt.ylabel('Accuracy (%)'); plt.title('Overfitting: Train vs Test')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('tmp_imgs/drybean_overfitting.png', dpi=100)

print(f"\n✓ 实验完成 | XGBoost是课上没讲过的算法（加分项）")
