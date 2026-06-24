"""
期末作业 Part 2: 数据处理（修复pandas COW问题）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

print("="*60)
print("Dry Bean — 数据处理")
print("="*60)

train = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_train.csv')
val = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_val.csv')
test = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset_Dirty_test.csv')

# 1. 类别清洗
clean_map = {
    'barbunya': 'BARBUNYA', 'bombay': 'BOMBAY', 'cali': 'CALI',
    'dermason': 'DERMASON', 'D3RMAS0N': 'DERMASON',
    'horoz': 'HOROZ', 'H0R0Z': 'HOROZ',
    'seker': 'SEKER', 'S3K3R': 'SEKER',
    'sira': 'SIRA', 'B0MBAY': 'BOMBAY',
}
for df in [train, val, test]:
    df['Class'] = df['Class'].str.strip().replace(clean_map)

# 2. 字符串特征转数值
for c in ['Solidity', 'Compactness']:
    for df in [train, val, test]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# 3. 缺失值填充（非inplace方式）
all_data = pd.concat([train[[c for c in train.columns if c != 'Class']],
                       val[[c for c in val.columns if c != 'Class']],
                       test[[c for c in test.columns if c != 'Class']]])
for col in all_data.columns:
    if all_data[col].isna().sum() > 0:
        med = all_data[col].median()
        train[col] = train[col].fillna(med)
        val[col] = val[col].fillna(med)
        test[col] = test[col].fillna(med)

print(f"清洗后: train={len(train)}  val={len(val)}  test={len(test)}")
print(f"NaN检查: train={train.isnull().sum().sum()}  val={val.isnull().sum().sum()}  test={test.isnull().sum().sum()}")

# 4. 特征缩放
num_cols = [c for c in train.columns if c != 'Class']
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
val[num_cols] = scaler.transform(val[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# 5. 保存
train.to_csv('DryBeanDataset/train_clean.csv', index=False)
val.to_csv('DryBeanDataset/val_clean.csv', index=False)
test.to_csv('DryBeanDataset/test_clean.csv', index=False)
import joblib; joblib.dump(scaler, 'models/drybean_scaler.pkl')
print("已保存清洗后数据 + scaler")
