import pandas as pd
import numpy as np


def preprocess_titanic(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 目标列名
    possible_targets = ["Survived", "2urvived", "survived"]
    target_col = None
    for col in possible_targets:
        if col in train_df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError("未找到目标列，请检查数据集中的标签列名。")

    # 删除无用列
    drop_cols = [col for col in train_df.columns if col.lower().startswith("zero")]
    for col in ["Passengerid", "PassengerId", "Name", "Ticket", "Cabin"]:
        if col in train_df.columns:
            drop_cols.append(col)

    train_df.drop(columns=list(set(drop_cols)), inplace=True, errors="ignore")
    test_df.drop(columns=list(set(drop_cols)), inplace=True, errors="ignore")

    # 类别型特征编码
    category_cols = []
    for col in train_df.columns:
        if col != target_col and train_df[col].dtype == "object":
            category_cols.append(col)

    # 拼接后统一做编码，保证 train/test 一致
    combined = pd.concat(
        [train_df.drop(columns=[target_col]), test_df.drop(columns=[target_col])],
        axis=0
    )

    # 缺失值填补
    for col in combined.columns:
        if combined[col].dtype == "object":
            mode_val = combined[col].mode()[0]
            combined[col] = combined[col].fillna(mode_val)
        else:
            mean_val = combined[col].mean()
            combined[col] = combined[col].fillna(mean_val)

    combined = pd.get_dummies(combined, columns=category_cols, drop_first=True)

    # 拆分回来
    X_train = combined.iloc[:len(train_df), :].values.astype(float)
    X_test = combined.iloc[len(train_df):, :].values.astype(float)

    y_train = train_df[target_col].values.astype(int)
    y_test = test_df[target_col].values.astype(int)

    # 标准化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test, mean, std
