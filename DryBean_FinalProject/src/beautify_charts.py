"""
美化期末大作业全部15张图表
统一风格: 深色标题、柔和配色、清晰标注
"""
import sys, pandas as pd, numpy as np, matplotlib, warnings
sys.stdout.reconfigure(encoding='utf-8')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# ===== 统一风格设置 =====
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DEE2E6',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#DEE2E6',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
VIVID_RED = '#E74C3C'
VIVID_BLUE = '#2C3E6B'

out = 'tmp_imgs'
train = pd.read_csv('DryBeanDataset/train_clean.csv')
train_raw = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_train.csv')
feat = [c for c in train.columns if c != 'Class']
classes = train['Class'].unique()
n_cls = len(classes)
palette = {cls: COLORS[i % len(COLORS)] for i, cls in enumerate(sorted(classes))}
le = LabelEncoder(); y = le.fit_transform(train['Class'])
X = train[feat].values

def save(fig, name):
    fig.savefig(f'{out}/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {name}.png')

# ================================================================
# 1. 数据清洗前后对比
# ================================================================
print('\n[1/15] 数据清洗对比...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

raw_counts = train_raw['Class'].value_counts()
clean_counts = train['Class'].value_counts()

bars1 = ax1.barh(range(len(raw_counts)), raw_counts.values, color='#FF6B6B', edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(raw_counts)))
ax1.set_yticklabels(raw_counts.index, fontsize=8)
ax1.set_title('Before Cleaning', fontsize=16, fontweight='bold', color=VIVID_BLUE, pad=15)
ax1.set_xlabel('Count', fontsize=12)
ax1.text(0.95, 0.95, f'{len(raw_counts)} classes', transform=ax1.transAxes,
         ha='right', va='top', fontsize=11, color='#FF6B6B', fontweight='bold')
for bar, v in zip(bars1, raw_counts.values):
    ax1.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2, str(v),
             va='center', fontsize=7, color='#666')

bars2 = ax2.barh(range(len(clean_counts)), clean_counts.values, color='#45B7D1', edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(clean_counts)))
ax2.set_yticklabels(clean_counts.index, fontsize=10)
ax2.set_title('After Cleaning', fontsize=16, fontweight='bold', color=VIVID_BLUE, pad=15)
ax2.set_xlabel('Count', fontsize=12)
ax2.text(0.95, 0.95, f'{len(clean_counts)} classes', transform=ax2.transAxes,
         ha='right', va='top', fontsize=11, color='#45B7D1', fontweight='bold')
for bar, v in zip(bars2, clean_counts.values):
    ax2.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2, str(v),
             va='center', fontsize=8, color='#666')

fig.suptitle('Data Cleaning Effect: Class Labels', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'drybean_data_cleaning')

# ================================================================
# 2. 类别分布
# ================================================================
print('[2/15] 类别分布...')
fig, ax = plt.subplots(figsize=(11, 6))
vc = train['Class'].value_counts().sort_index()
from matplotlib import cm
gradients = [plt.cm.RdYlBu(i/len(vc)) for i in range(len(vc))]
bars = ax.bar(range(len(vc)), vc.values, color=[palette[c] for c in vc.index],
              edgecolor='white', linewidth=1.5, width=0.7, alpha=0.9)
# 添加渐变底纹
for bar, c in zip(bars, [palette[c] for c in vc.index]):
    ax.plot([bar.get_x(), bar.get_x()+bar.get_width()], [bar.get_height()]*2,
            color=c, linewidth=3, alpha=0.6)
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(vc.index, fontsize=11, fontweight='bold')
ax.set_ylabel('Count', fontsize=13, fontweight='bold')
ax.set_title('Class Distribution in Training Set', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
# 顶部标注
for bar, v in zip(bars, vc.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+25, str(v),
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=VIVID_BLUE)
# 百分比标注在柱内
for bar, v in zip(bars, vc.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2, f'{v/len(train)*100:.1f}%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
ax.set_ylim(0, max(vc.values)*1.15)
save(fig, 'drybean_class_dist')

# ================================================================
# 3. 特征相关矩阵
# ================================================================
print('[3/15] 特征相关矩阵...')
fig, ax = plt.subplots(figsize=(11, 9))
corr = train[feat].corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(range(len(feat)))
ax.set_yticks(range(len(feat)))
ax.set_xticklabels(feat, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(feat, fontsize=8)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', color=VIVID_BLUE, pad=15)
# 标注数值
for i in range(len(feat)):
    for j in range(len(feat)):
        val = corr.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else '#333'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
plt.colorbar(im, ax=ax, shrink=0.8)
save(fig, 'drybean_correlation')

# ================================================================
# 4. 特征盒图（选6个主要特征）
# ================================================================
print('[4/15] 特征盒图...')
main_feat = ['Area', 'Perimeter', 'MajorAxisLength', 'AspectRation', 'Eccentricity', 'roundness']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.flat, main_feat):
    data = [train[train['Class']==c][col].dropna().values for c in sorted(classes)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp['boxes'], [palette[c] for c in sorted(classes)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticklabels(sorted(classes), rotation=30, fontsize=7)
    ax.set_title(col, fontsize=13, fontweight='bold', color=VIVID_BLUE)
    ax.set_ylabel('Value (standardized)', fontsize=9)
fig.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'drybean_boxplot')

# ================================================================
# 5. PCA 可视化
# ================================================================
print('[5/15] PCA...')
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(11, 9))
for cls in sorted(classes):
    mask = train['Class'] == cls
    ax.scatter(Xp[mask, 0], Xp[mask, 1], c=[palette[cls]], label=cls,
               alpha=0.5, s=12, edgecolors='white', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13, fontweight='bold')
ax.set_title('PCA: 2D Projection of Dry Bean Features', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
ax.legend(fontsize=10, markerscale=3, framealpha=0.9, edgecolor='#999', loc='best')
save(fig, 'drybean_pca')

# ================================================================
# 6. 特征重要性
# ================================================================
print('[6/15] 特征重要性...')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
imp = pd.DataFrame({'feature': feat, 'importance': rf.feature_importances_}).sort_values('importance')
fig, ax = plt.subplots(figsize=(10, 7))
bar_colors = [plt.cm.Set2(i/len(imp)) for i in range(len(imp))]
bars = ax.barh(imp['feature'], imp['importance'], color=bar_colors,
               edgecolor='white', linewidth=1.5)
ax.set_xlabel('Importance', fontsize=13, fontweight='bold')
ax.set_title('Feature Importance (Random Forest)', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
for bar, v in zip(bars, imp['importance']):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=10, fontweight='bold', color='#555')
save(fig, 'drybean_feature_importance')

# ================================================================
# 7. 混淆矩阵
# ================================================================
print('[7/15] 混淆矩阵...')
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)
yp = lr.predict(X)
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y, yp)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d', colorbar=True, im_kw={'vmin':0, 'vmax':cm.max()*1.1})
for text in ax.texts:
    text.set_fontsize(11)
    try:
        v = int(text.get_text())
        text.set_color('#2C3E6B' if v < cm.max()*0.4 else 'white')
    except:
        pass
ax.set_title('Confusion Matrix (Logistic Regression)', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
save(fig, 'drybean_confusion_matrix')

# ================================================================
# 8. 准确率对比
# ================================================================
print('[8/15] 准确率对比...')
fig, ax = plt.subplots(figsize=(12, 7))
algos = ['Logistic\nRegression', 'SVM\n(RBF)', 'KNN\n(k=5)', 'XGBoost']
train_acc = [92.51, 92.99, 93.78, 99.58]
val_acc = [92.35, 92.95, 91.69, 92.65]
test_acc = [91.85, 92.88, 91.74, 92.58]
x = np.arange(len(algos)); w = 0.22
for ref in [92, 95, 98]:
    ax.axhline(y=ref, color='#e8e8e8', linestyle='--', linewidth=1, zorder=0)
b1 = ax.bar(x-w, train_acc, w, label='Train', color='#FF6B6B', edgecolor='white', linewidth=1.5, alpha=0.9)
b2 = ax.bar(x, val_acc, w, label='Validation', color='#4ECDC4', edgecolor='white', linewidth=1.5, alpha=0.9)
b3 = ax.bar(x+w, test_acc, w, label='Test', color='#2E86DE', edgecolor='white', linewidth=1.5, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Algorithm Accuracy Comparison', fontsize=20, fontweight='bold', color=VIVID_BLUE, pad=18)
ax.set_ylim(85, 102); ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
for bar in b3:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{bar.get_height():.2f}%',
            ha='center', fontsize=11, fontweight='bold', color='#2E86DE')
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{bar.get_height():.2f}%',
            ha='center', fontsize=9, color='#FF6B6B', fontweight='bold')
# Best marker arrow
best_idx = np.argmax(test_acc)
ax.annotate('★ Best {:.2f}%'.format(max(test_acc)), xy=(best_idx+w, max(test_acc)),
            xytext=(best_idx+w+0.5, max(test_acc)+3),
            fontsize=13, color='#E74C3C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', linewidth=2))
save(fig, 'drybean_accuracy')

# ================================================================
# 9. 推理速度对比
# ================================================================
print('[9/15] 推理速度...')
fig, ax = plt.subplots(figsize=(11, 6))
times_ms = [6.0, 4526, 785, 116]
speed_colors = ['#2ECC71', '#E74C3C', '#F39C12', '#3498DB']
bars = ax.barh(algos, times_ms, color=speed_colors, edgecolor='white', linewidth=2, height=0.6)
ax.set_xlabel('Inference Time (ms) - log scale', fontsize=13, fontweight='bold')
ax.set_title('Inference Speed Comparison', fontsize=20, fontweight='bold', color=VIVID_BLUE, pad=18)
ax.set_xscale('log')
ax.set_xlim(3, 10000)
for bar, v, c in zip(bars, times_ms, speed_colors):
    ax.text(bar.get_width()*1.08, bar.get_y()+bar.get_height()/2,
            f'{v:.0f}ms' if v>1 else f'{v:.1f}ms', va='center', fontsize=12, fontweight='bold', color=c)
ratings = ['🚀 Fastest', '🐢 Slowest', '🐇 Fast', '⚡ Quick']
for bar, v, r in zip(bars, times_ms, ratings):
    ax.text(bar.get_width()/2, bar.get_y()+bar.get_height()/2, r,
            va='center', ha='center', fontsize=11, fontweight='bold', color='white', alpha=0.9)
ax.text(0.98, 0.02, f'SVM is {4526/6:.0f}x slower than LR', transform=ax.transAxes,
        ha='right', fontsize=11, color='#E74C3C', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD', alpha=0.8))
save(fig, 'drybean_speed')

# ================================================================
# 10. 鲁棒性曲线
# ================================================================
print('[10/15] 鲁棒性...')
fig, ax = plt.subplots(figsize=(10, 6))
noise_vals = [0.0, 0.1, 0.2, 0.5, 1.0]
rob_data = {
    'Logistic Regression': [91.9, 91.9, 89.3, 79.4, 60.2],
    'SVM (RBF)': [92.9, 92.5, 91.0, 85.5, 68.2],
    'KNN (k=5)': [91.7, 91.4, 90.7, 84.8, 70.2],
    'XGBoost': [92.6, 91.5, 88.8, 75.2, 54.6],
}
line_styles = ['o-', 's-', 'D-', '^-']
line_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
for (name, vals), ls, c in zip(rob_data.items(), line_styles, line_colors):
    ax.plot(noise_vals, vals, ls, label=name, color=c, linewidth=2.5, markersize=8)
ax.set_xlabel('Noise Level (σ)', fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Robustness: Accuracy vs Noise Level', fontsize=16, fontweight='bold', color=VIVID_BLUE)
ax.legend(fontsize=11, framealpha=0.8)
ax.grid(True, alpha=0.3)
save(fig, 'drybean_robustness')

# ================================================================
# 11. 过拟合分析
# ================================================================
print('[11/15] 过拟合...')
fig, ax = plt.subplots(figsize=(10, 6))
algo_labels = ['Logistic\nRegression', 'SVM\n(RBF)', 'KNN\n(k=5)', 'XGBoost']
t_acc = [92.51, 92.99, 93.78, 99.58]
e_acc = [91.85, 92.88, 91.74, 92.58]
diffs = [t-e for t,e in zip(t_acc, e_acc)]
x = np.arange(len(algo_labels)); w = 0.35
b1 = ax.bar(x-w/2, t_acc, w/2, label='Train', color='#FF6B6B', edgecolor='white')
b2 = ax.bar(x+w/2, e_acc, w/2, label='Test', color='#45B7D1', edgecolor='white')
ax2 = ax.twinx()
ax2.plot(x, diffs, 'r--o', linewidth=2, markersize=8, color='#96CEB4', label='Gap')
ax2.set_ylabel('Train-Test Gap (%)', fontsize=13, color='#C44E52')
ax2.tick_params(axis='y', colors='#C44E52')
for i, d in enumerate(diffs):
    ax2.text(i, d+0.3, f'{d:.2f}%', ha='center', fontsize=10, color='#96CEB4', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(algo_labels, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Overfitting Analysis: Train vs Test Gap', fontsize=16, fontweight='bold', color=VIVID_BLUE)
ax.legend(loc='lower center', fontsize=10)
save(fig, 'drybean_overfitting')

# ================================================================
# 12. XGBoost 学习曲线
# ================================================================
print('[12/15] XGBoost学习曲线...')
X_tr, y_tr = X[:8000], y[:8000]
X_va, y_va = X[8000:], y[8000:]
train_s, val_s = [], []
for n in range(10, 201, 10):
    xgb = XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    train_s.append(accuracy_score(y_tr, xgb.predict(X_tr))*100)
    val_s.append(accuracy_score(y_va, xgb.predict(X_va))*100)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(10, 201, 10), train_s, 'o-', color='#FF6B6B', label='Train', linewidth=2, markersize=5)
ax.plot(range(10, 201, 10), val_s, 's-', color='#4ECDC4', label='Validation', linewidth=2, markersize=5)
best_n = range(10, 201, 10)[np.argmax(val_s)]
best_v = max(val_s)
ax.axvline(x=best_n, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'Best: n={best_n}, acc={best_v:.1f}%',
            xy=(best_n, best_v), xytext=(best_n+20, best_v-2),
            fontsize=11, color='#96CEB4', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#C44E52'))
ax.set_xlabel('n_estimators', fontsize=13); ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('XGBoost Learning Curve', fontsize=16, fontweight='bold', color=VIVID_BLUE)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
save(fig, 'drybean_xgb_learning_curve')

# ================================================================
# 13. 特征散点图
# ================================================================
print('[13/15] 特征散点图...')
top3_idx = np.argsort(rf.feature_importances_)[-3:]
top3_feat = [feat[i] for i in top3_idx]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
pairs = [(top3_feat[0], top3_feat[1]), (top3_feat[0], top3_feat[2]), (top3_feat[1], top3_feat[2])]
for ax, (f1, f2) in zip(axes, pairs):
    for cls in sorted(classes):
        mask = train['Class'] == cls
        ax.scatter(train.loc[mask, f1], train.loc[mask, f2], c=[palette[cls]],
                   label=cls, alpha=0.4, s=8, edgecolors='white', linewidth=0.2)
    ax.set_xlabel(f1, fontsize=11); ax.set_ylabel(f2, fontsize=11)
    ax.legend(fontsize=7, markerscale=3, framealpha=0.7)
    ax.set_title(f'{f1} vs {f2}', fontsize=13, fontweight='bold', color=VIVID_BLUE)
fig.suptitle('Feature Pair Scatter Plots', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'drybean_scatter_pairs')

# ================================================================
# 14. 噪声影响示例
# ================================================================
print('[14/15] 噪声影响示例...')
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
np.random.seed(42)
X_sample = X[:1]
noises = [0.0, 0.5, 1.0]
for ax, noise, title in zip(axes, noises, ['Original (σ=0)', 'Moderate Noise (σ=0.5)', 'Heavy Noise (σ=1.0)']):
    Xn = X_sample + np.random.randn(*X_sample.shape) * noise
    ax.bar(range(len(feat)), X_sample[0], color='#2E86DE', alpha=0.9, label='Original', width=0.4, edgecolor='white')
    ax.bar(range(len(feat)), Xn[0], color='#E74C3C', alpha=0.7, label='Noisy', width=0.4, edgecolor='#333', linewidth=0.3)
    ax.set_xticks(range(len(feat)))
    ax.set_xticklabels(feat, rotation=45, ha='right', fontsize=6)
    ax.set_title(title, fontsize=13, fontweight='bold', color=VIVID_BLUE, pad=10)
    ax.set_ylabel('Feature Value')
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
fig.suptitle('Effect of Gaussian Noise on Feature Values', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'drybean_noise_example')

# ================================================================
# 15. 特征直方图（部分）
# ================================================================
print('[15/15] 特征直方图...')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.flat, main_feat):
    for cls in sorted(classes):
        d = train[train['Class']==cls][col].dropna()
        ax.hist(d, bins=40, alpha=0.4, color=palette[cls], label=cls, density=True)
    ax.set_title(col, fontsize=13, fontweight='bold', color=VIVID_BLUE)
    ax.tick_params(labelsize=8)
    if ax in [axes.flat[-1]]:
        ax.legend(fontsize=7, framealpha=0.7)
fig.suptitle('Feature Distributions by Class (Density)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'drybean_feature_histograms')

print(f'\n🎨 全部15张图表已美化完成! 保存到 {out}/')
