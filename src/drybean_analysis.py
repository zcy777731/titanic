"""
期末作业 Part 1: 数据分析
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns

print("="*60)
print("Dry Bean Dataset — 数据分析")
print("="*60)

# 1. 加载
train = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_train.csv')
val = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_val.csv')
test = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_test.csv')
print(f"训练: {len(train)}  验证: {len(val)}  测试: {len(test)}  总计: {len(train)+len(val)+len(test)}")

num_cols = [c for c in train.columns if c != 'Class']
str_cols = [c for c in num_cols if train[c].dtype == object]
print(f"\n特征 ({len(num_cols)}个): {num_cols}")
print(f"字符串类型特征（含数据污染）: {str_cols}")

# 2. 数据污染
print("\n" + "="*60)
print("数据污染分析")
print("="*60)

# 2.1 类别标签
all_cls = pd.concat([train['Class'], val['Class'], test['Class']])
clean_map = {
    'barbunya': 'BARBUNYA', 'bombay': 'BOMBAY', 'cali': 'CALI',
    'dermason': 'DERMASON', 'D3RMAS0N': 'DERMASON',
    'horoz': 'HOROZ', 'H0R0Z': 'HOROZ',
    'seker': 'SEKER', 'S3K3R': 'SEKER',
    'sira': 'SIRA', 'B0MBAY': 'BOMBAY',
}
# 尾部带空格的也处理
for c in all_cls.unique():
    if c.strip() != c:
        clean_map[c] = c.strip()
        print(f"  ⚠ 尾部空格: '{c}' -> '{c.strip()}'")
dirty_cls = [c for c in all_cls.unique() if c not in ['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA'] and c not in clean_map]
print(f"\n类别污染: 标准7类, 实际{len(all_cls.unique())}种")
for c in sorted(dirty_cls):
    print(f"  ⚠ '{c}' -> '{clean_map.get(c, c.strip())}' ({all_cls.value_counts()[c]}条)")
for c in ['dermason', 'D3RMAS0N', 'sira', 'seker', 'S3K3R', 'horoz', 'H0R0Z', 'barbunya', 'bombay', 'cali', 'B0MBAY']:
    print(f"  ⚠ '{c}' -> '{clean_map[c]}' ({all_cls.value_counts()[c]}条)")

# 2.2 缺失值
print(f"\n缺失值:")
for n, df in [('train',train),('val',val),('test',test)]:
    print(f"  {n}: {df.isnull().sum().sum()}/{df.size} ({df.isnull().sum().sum()/df.size*100:.2f}%)")

# 2.3 字符串特征污染
print(f"\n字符串特征污染:")
for c in str_cols:
    numeric = pd.to_numeric(train[c], errors='coerce')
    bad = numeric.isna()
    print(f"  {c}: {bad.sum()}个非数值 ({train[c][bad].unique() if bad.sum()>0 else '无'})")

# 2.4 异常值
print(f"\n异常值 (IQR×3):")
num_cols_clean = train.select_dtypes(include=[np.number]).columns
for col in num_cols_clean:
    Q1, Q3 = train[col].quantile(0.25), train[col].quantile(0.75)
    IQR = Q3 - Q1
    n = ((train[col] < Q1-3*IQR) | (train[col] > Q3+3*IQR)).sum()
    if n > 0:
        print(f"  {col}: {n}个")

# 3. 类别分布
print(f"\n" + "="*60)
print("清洗后类别分布")
print("="*60)
for n, df in [('train',train),('val',val),('test',test)]:
    df['Class'] = df['Class'].str.strip().replace(clean_map)
    print(f"\n{n}:")
    for cls, cnt in df['Class'].value_counts().sort_index().items():
        print(f"  {cls:12s}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")

# 4. 特征统计
print(f"\n" + "="*60)
print("特征统计")
print("="*60)
for c in num_cols_clean:
    print(f"  {c:20s}  mean={train[c].mean():10.2f}  std={train[c].std():10.2f}  min={train[c].min():10.2f}  max={train[c].max():10.2f}")

# 5. 可视化
print(f"\n生成可视化...")
train['Class'] = train['Class'].str.strip().replace(clean_map)

# 5.1 类别分布
fig, ax = plt.subplots(figsize=(10, 5))
vc = train['Class'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(vc)))
ax.bar(vc.index, vc.values, color=colors, edgecolor='gray')
for i, v in enumerate(vc.values):
    ax.text(i, v+20, str(v), ha='center', fontsize=10)
ax.set_title(f'Training Set Class Distribution (n={len(train)})')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('tmp_imgs/drybean_class_dist.png', dpi=100)
print(f"  类别分布: tmp_imgs/drybean_class_dist.png")

# 5.2 相关矩阵
plt.figure(figsize=(10, 8))
corr = train[num_cols_clean].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('tmp_imgs/drybean_correlation.png', dpi=100)
print(f"  相关矩阵: tmp_imgs/drybean_correlation.png")

# 5.3 盒图
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
for ax, col in zip(axes.flat, num_cols_clean):
    train.boxplot(column=col, by='Class', ax=ax, fontsize=8)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
plt.suptitle('Feature Distributions by Class', fontsize=14)
plt.tight_layout()
plt.savefig('tmp_imgs/drybean_boxplot.png', dpi=100)
print(f"  特征分布盒图: tmp_imgs/drybean_boxplot.png")

print(f"\n✓ 数据分析完成")
print(f"主要发现:")
print(f"  1. 类别标签污染: 大小写、拼写错误(0→O, S→3)、尾部空格")
print(f"  2. 特征值污染: Solidify/Compactness含非数值")
print(f"  3. 缺失值: 约0.5%")
print(f"  4. 类别不平衡: DERMASON最多(25%), BOMBAY最少(3.5%)")
print(f"  5. 特征高相关: Area/Perimeter/ConvexArea等高度相关")
