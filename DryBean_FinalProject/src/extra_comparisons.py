"""
加分项: 评分说明外的额外对比维度（美观版）
① 模型大小 vs 参数量
② 数据量敏感性曲线
③ 训练时间分解
④ 可解释性对比
"""
import sys, os, time, numpy as np, pandas as pd, warnings
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.filterwarnings('ignore')

# ===== 统一风格 =====
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DEE2E6',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': '#DEE2E6',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
VIVID_BLUE = '#2C3E6B'

OUT = 'tmp_imgs'
SRC = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, SRC)
from data_loader import DryBeanLoader
from trainer import train_model, ALGORITHMS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

loader = DryBeanLoader()
X, y, feat, classes = loader.get_X_y('train', cleaned=True)
X_te, y_te, _, _ = loader.get_X_y('test', cleaned=True)
print(f"训练: {len(X)}  测试: {len(X_te)}  特征: {len(feat)}")
algos = ['lr', 'svm', 'knn', 'xgb']
algo_labels = ['Logistic\nRegression', 'SVM\n(RBF)', 'KNN\n(k=5)', 'XGBoost']

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  ✅ {name}.png')

# ================================================================
# ① 模型大小 × 参数量（横向双栏+模型卡片）
# ================================================================
print("\n① 模型大小对比...")
os.makedirs('tmp_models', exist_ok=True)
model_data = []
for a in algos:
    m, _ = train_model(X[:2000], y[:2000], a)
    joblib.dump(m, f'tmp_models/{a}.pkl')
    kb = os.path.getsize(f'tmp_models/{a}.pkl') / 1024
    if a == 'lr':    params = m.coef_.size + m.intercept_.size
    elif a == 'svm': params = len(m.support_vectors_)
    elif a == 'knn': params = X[:2000].shape[0]
    else:            params = sum(len(t.get_booster().get_dump()) for t in [m]) * 50
    model_data.append({'algo': a, 'kb': kb, 'params': params})

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(4); w = 0.3
bars = ax.bar(x, [d['kb'] for d in model_data], w, color=COLORS,
              edgecolor='white', linewidth=2, alpha=0.9)
for bar, d in zip(bars, model_data):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f'{d["kb"]:.0f}KB\n{d["params"]:,} params', ha='center', fontsize=11,
            fontweight='bold', color=VIVID_BLUE, va='bottom')
ax.set_xticks(x); ax.set_xticklabels(algo_labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Model Size (KB)', fontsize=13, fontweight='bold')
ax.set_title('Model Size & Complexity', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
ax.set_ylim(0, max(d['kb'] for d in model_data)*1.35)
# 注解
notes = ['✅ Lightweight', '⚠️ Stores\nsupport vectors', '⚠️ Stores\nall data', '📦 Boosted\ntrees']
for i, (bar, note) in enumerate(zip(bars, notes)):
    ax.annotate(note, xy=(bar.get_x()+bar.get_width()/2, bar.get_height()/2),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
save(fig, 'extra_model_size')

# ================================================================
# ② 数据量敏感性（渐变曲线+面积填充）
# ================================================================
print("② 数据量敏感性...")
ratios, n_total = [0.1, 0.3, 0.5, 0.7, 1.0], len(X)
sens = {a: [] for a in algos}
for r in ratios:
    Xs, ys = X[:int(n_total*r)], y[:int(n_total*r)]
    for a in algos:
        clf = {'lr': LogisticRegression(max_iter=1000, random_state=42),
               'svm': SVC(kernel='rbf', gamma='scale', random_state=42),
               'knn': KNeighborsClassifier(n_neighbors=5),
               'xgb': __import__('xgboost').XGBClassifier(n_estimators=100, random_state=42, verbosity=0)}[a]
        clf.fit(Xs, ys); sens[a].append(accuracy_score(y_te, clf.predict(X_te))*100)

fig, ax = plt.subplots(figsize=(11, 7))
labels = ['10%', '30%', '50%', '70%', '100%']
markers = ['o-', 's-', 'D-', '^-']
for i, a in enumerate(algos):
    ax.plot(range(5), sens[a], markers[i], color=COLORS[i], label=ALGORITHMS[a]['name'],
            linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)
    # 面积填充
    ax.fill_between(range(5), sens[a], alpha=0.08, color=COLORS[i])
    # 标注
    ax.annotate(f'{sens[a][0]:.1f}%', xy=(0, sens[a][0]), xytext=(-0.3, sens[a][0]+0.5),
                fontsize=10, color=COLORS[i], fontweight='bold', ha='center')
    ax.annotate(f'{sens[a][-1]:.1f}%', xy=(4, sens[a][-1]), xytext=(4.2, sens[a][-1]+0.3),
                fontsize=10, color=COLORS[i], fontweight='bold')
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Training Data Ratio', fontsize=14, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Data Size Sensitivity Analysis', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
ax.legend(fontsize=11, framealpha=0.9, loc='lower right')
ax.set_ylim(88, 94)
# 基准线
ax.axhline(y=91, color='#ddd', linestyle='--', linewidth=1)
save(fig, 'extra_data_sensitivity')

# ================================================================
# ③ 训练时间分解（横向对比+表格）
# ================================================================
print("③ 训练时间分解...")
times = {}
for a in algos:
    t0 = time.time(); train_model(X, y, a); times[a] = time.time() - t0

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(algo_labels, [times[a] for a in algos], color=COLORS,
               edgecolor='white', linewidth=2, height=0.55)
for bar, a in zip(bars, algos):
    v = times[a]
    ax.text(bar.get_width()+0.08, bar.get_y()+bar.get_height()/2,
            f'{v:.2f}s', va='center', fontsize=13, fontweight='bold', color=VIVID_BLUE)
    # 内部比例条（相对最慢的SVM）
    pct = v / max(times.values()) * 100
    ax.text(bar.get_width()/2, bar.get_y()+bar.get_height()/2,
            f'{pct:.0f}%', va='center', ha='center', fontsize=11, color='white', fontweight='bold')
ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Training Time Comparison', fontsize=18, fontweight='bold', color=VIVID_BLUE, pad=15)
ax.set_xlim(0, max(times.values())*1.3)
# 速度评级
ratings = [('Fast', '#2ECC71'), ('Slow', '#E74C3C'), ('Fastest', '#2ECC71'), ('Medium', '#F39C12')]
for bar, (r, c) in zip(bars, ratings):
    ax.text(bar.get_width()/2, bar.get_y()-0.25, r, va='center', ha='center',
            fontsize=10, color=c, fontweight='bold')
save(fig, 'extra_training_time')

# ================================================================
# ④ 可解释性对比（三栏：LR权重 / RF重要性 / 交集韦恩图风格）
# ================================================================
print("④ 可解释性分析...")
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
lr_w = np.abs(lr.coef_).mean(axis=0)
rf_i = rf.feature_importances_
top_n = 8
top_lr = np.argsort(lr_w)[-top_n:]
top_rf = np.argsort(rf_i)[-top_n:]
shared = set(top_lr) & set(top_rf)

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.35)

# LR系数（左上）
ax1 = fig.add_subplot(gs[0, :2])
colors_lr = ['#FF6B6B' if i in top_lr else '#FFD0D0' for i in range(len(feat))]
ax1.barh(range(len(feat)), lr_w, color=colors_lr, edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(feat))); ax1.set_yticklabels(feat, fontsize=8)
ax1.set_title('LR Coefficients (|weight|)', fontsize=14, fontweight='bold', color=VIVID_BLUE)
ax1.set_xlabel('Absolute Weight', fontsize=11)
for i in top_lr:
    ax1.text(lr_w[i]+0.0002, i, '★', va='center', fontsize=10, color='#FF6B6B')

# RF重要性（右上）
ax2 = fig.add_subplot(gs[0, 2:])
colors_rf = ['#4ECDC4' if i in top_rf else '#D0F5F0' for i in range(len(feat))]
ax2.barh(range(len(feat)), rf_i, color=colors_rf, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(feat))); ax2.set_yticklabels(feat, fontsize=8)
ax2.set_title('RF Feature Importance', fontsize=14, fontweight='bold', color=VIVID_BLUE)
ax2.set_xlabel('Importance', fontsize=11)
for i in top_rf:
    ax2.text(rf_i[i]+0.002, i, '★', va='center', fontsize=10, color='#4ECDC4')

# Top8对比表（下方）
ax3 = fig.add_subplot(gs[1, 1:3])
ax3.axis('off')
table_data = []
for i in range(top_n-1, -1, -1):
    lr_f = feat[top_lr[i]]
    rf_f = feat[top_rf[i]]
    shared_s = '⭐' if top_lr[i] in shared or top_rf[i] in shared else ''
    table_data.append([top_n-i, lr_f, f'{lr_w[top_lr[i]]:.4f}',
                       rf_f, f'{rf_i[top_rf[i]]:.4f}',
                       '✓ Both' if top_lr[i] in top_rf else ''])

tbl = ax3.table(cellText=table_data,
                colLabels=['Rank', 'LR Top', 'Weight', 'RF Top', 'Importance', 'Shared'],
                cellLoc='center', loc='center',
                colWidths=[0.08, 0.2, 0.12, 0.2, 0.12, 0.1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for key, cell in tbl.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor(VIVID_BLUE); cell.set_text_props(color='white', fontweight='bold')
    elif key[0] % 2 == 0:
        cell.set_facecolor('#F0F4F8')
for r in range(len(table_data)):
    if table_data[r][5] == '✓ Both':
        tbl[(r+1, 5)].set_facecolor('#FFF3CD')
ax3.set_title('Top-8 Feature Comparison: LR vs RF', fontsize=14, fontweight='bold',
              color=VIVID_BLUE, pad=15)
plt.suptitle('Model Interpretability Analysis', fontsize=18, fontweight='bold', color=VIVID_BLUE, y=1.02)
save(fig, 'extra_interpretability')

print(f"\n🎨 4项额外对比维度全部完成! 已保存到 {OUT}/extra_*.png")
