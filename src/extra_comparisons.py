"""
加分项: 除评分说明外的额外对比维度
① 模型大小对比（文件大小/参数量）
② 数据量敏感性（训练比例 vs 准确率）
③ 训练时间分解（分阶段耗时）
④ 可解释性对比（LR系数 vs 树模型重要特征交集）
"""
import sys, os, time, numpy as np, pandas as pd, matplotlib, warnings
sys.stdout.reconfigure(encoding='utf-8')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

OUT = 'tmp_imgs'
SRC = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, SRC)
from data_loader import DryBeanLoader
from trainer import train_model, ALGORITHMS

loader = DryBeanLoader()
X, y, feat, classes = loader.get_X_y('train', cleaned=True)
X_te, y_te, _, _ = loader.get_X_y('test', cleaned=True)
print(f"训练: {len(X)}  测试: {len(X_te)}  特征: {len(feat)}")

# ================================================================
# ① 模型大小对比
# ================================================================
print("\n[1/4] 模型大小对比...")
import joblib

model_sizes = {}
for algo in ['lr', 'svm', 'knn', 'xgb']:
    model, _ = train_model(X[:2000], y[:2000], algo)  # 用小样本加速
    joblib.dump(model, f'tmp_models/{algo}.pkl')
    size_kb = os.path.getsize(f'tmp_models/{algo}.pkl') / 1024
    # 参数量估计
    if algo == 'lr':
        params = model.coef_.size + model.intercept_.size
    elif algo == 'svm':
        params = len(model.support_vectors_) * X.shape[1] + len(model.intercept_)
    elif algo == 'knn':
        params = X[:2000].size  # 存储所有训练数据
    elif algo == 'xgb':
        params = sum(len(t.get_booster().get_dump()) for t in [model]) * 100
    model_sizes[algo] = {'size_kb': size_kb, 'params': params}

# 图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
algos_label = ['Logistic\nRegression', 'SVM\n(RBF)', 'KNN\n(k=5)', 'XGBoost']
sizes = [model_sizes[a]['size_kb'] for a in ['lr','svm','knn','xgb']]
param_counts = [model_sizes[a]['params'] for a in ['lr','svm','knn','xgb']]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

axes[0].bar(algos_label, sizes, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('Model Size (KB)', fontsize=12, fontweight='bold')
axes[0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
for i, v in enumerate(sizes):
    axes[0].text(i, v+0.5, f'{v:.0f}KB', ha='center', fontsize=10, fontweight='bold')

axes[1].bar(algos_label, param_counts, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_ylabel('Parameter Count (est.)', fontsize=12, fontweight='bold')
axes[1].set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
for i, v in enumerate(param_counts):
    axes[1].text(i, v+max(param_counts)*0.01, f'{v:,}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/extra_model_size.png', dpi=150)
plt.close()
print(f"  ✅ extra_model_size.png")

# ================================================================
# ② 数据量敏感性
# ================================================================
print("[2/4] 数据量敏感性分析...")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

ratios = [0.1, 0.3, 0.5, 0.7, 1.0]
n_total = len(X)
sensitivity = {algo: [] for algo in ['lr', 'svm', 'knn', 'xgb']}

for ratio in ratios:
    n_use = int(n_total * ratio)
    X_sub, y_sub = X[:n_use], y[:n_use]
    for algo in ['lr', 'svm', 'knn', 'xgb']:
        if algo == 'lr':
            m = LogisticRegression(max_iter=1000, random_state=42)
        elif algo == 'svm':
            m = SVC(kernel='rbf', gamma='scale', random_state=42)
        elif algo == 'knn':
            m = KNeighborsClassifier(n_neighbors=5)
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        m.fit(X_sub, y_sub)
        acc = accuracy_score(y_te, m.predict(X_te))
        sensitivity[algo].append(acc)

fig, ax = plt.subplots(figsize=(10, 6))
markers = ['o-', 's-', 'D-', '^-']
for i, algo in enumerate(['lr', 'svm', 'knn', 'xgb']):
    ax.plot([f'{int(r*100)}%' for r in ratios], [a*100 for a in sensitivity[algo]],
            markers[i], label=ALGORITHMS[algo]['name'], color=colors[i], linewidth=2.5, markersize=8)
ax.set_xlabel('Training Data Ratio', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Data Size Sensitivity', fontsize=15, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/extra_data_sensitivity.png', dpi=150)
plt.close()
print(f"  ✅ extra_data_sensitivity.png")
for r, ratio in enumerate(ratios):
    print(f"    {int(ratio*100):3d}% data: ", end='')
    for algo in ['lr', 'svm', 'knn', 'xgb']:
        print(f"{algo}={sensitivity[algo][r]*100:.1f}% ", end='')
    print()

# ================================================================
# ③ 训练时间分解（分阶段）
# ================================================================
print("[3/4] 训练时间分解...")

time_decomp = {}
for algo in ['lr', 'svm', 'knn', 'xgb']:
    t0 = time.time()
    model, _ = train_model(X, y, algo)
    total = time.time() - t0
    time_decomp[algo] = total

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(algos_label, [time_decomp[a] for a in ['lr','svm','knn','xgb']],
        color=colors, edgecolor='white', linewidth=1.5, height=0.6)
ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
for i, a in enumerate(['lr','svm','knn','xgb']):
    v = time_decomp[a]
    ax.text(v+0.1, i, f'{v:.2f}s', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/extra_training_time.png', dpi=150)
plt.close()
print(f"  ✅ extra_training_time.png")

# ================================================================
# ④ 可解释性对比
# ================================================================
print("[4/4] 可解释性分析...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# LR系数
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)
lr_weights = np.abs(lr.coef_).mean(axis=0)

# RF特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 排序
top_n = 8
lr_top_idx = np.argsort(lr_weights)[-top_n:]
rf_top_idx = np.argsort(rf.feature_importances_)[-top_n:]
overlap = set(lr_top_idx) & set(rf_top_idx)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# LR系数
axes[0].barh(range(top_n), lr_weights[lr_top_idx][::-1], color='#FF6B6B', edgecolor='white')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels([feat[i] for i in lr_top_idx[::-1]], fontsize=9)
axes[0].set_title('LR Feature Weights (|coef|)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Absolute Weight')

# RF重要性
axes[1].barh(range(top_n), rf.feature_importances_[rf_top_idx][::-1], color='#4ECDC4', edgecolor='white')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels([feat[i] for i in rf_top_idx[::-1]], fontsize=9)
axes[1].set_title('RF Feature Importance', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Importance')

# 交集分析
overlap_names = [feat[i] for i in overlap]
all_top = sorted(set(lr_top_idx) | set(rf_top_idx))
colors_venn = ['#FF6B6B' if i in lr_top_idx else 'white' for i in all_top]
colors_venn2 = ['#4ECDC4' if i in rf_top_idx else 'white' for i in all_top]
axes[2].axis('off')
axes[2].text(0.5, 0.9, 'Shared Top Features', ha='center', fontsize=13, fontweight='bold')
for i, idx in enumerate(all_top):
    in_lr = '✓' if idx in lr_top_idx else ' '
    in_rf = '✓' if idx in rf_top_idx else ' '
    shared = '⭐' if idx in overlap else ' '
    axes[2].text(0.5, 0.75-i*0.08, f'{feat[idx]:25s}  LR:{in_lr}  RF:{in_rf}  {shared}',
                 ha='center', fontsize=10, transform=axes[2].transAxes)
axes[2].text(0.5, 0.03, f'{len(overlap)}/{top_n} features shared', ha='center',
             fontsize=11, transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig(f'{OUT}/extra_interpretability.png', dpi=150)
plt.close()
print(f"  ✅ extra_interpretability.png")

print(f"\n{'='*50}")
print(f"4项额外对比维度全部完成!")
print(f"{'='*50}")
print(f"图表保存: {OUT}/extra_*.png")
