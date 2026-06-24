"""
数据加载与预处理模块
支持: Dry Bean Dataset
功能: 加载原始数据 / 清洗 / 特征缩放 / 保存
"""
import sys, pandas as pd, numpy as np, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os

class DryBeanLoader:
    """干豆数据集加载器"""

    def __init__(self, raw_dir='DryBeanDataset', clean_dir='DryBeanDataset'):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.scaler = None
        self.label_encoder = None
        self.feat_cols = None
        self.clean_map = {
            'barbunya': 'BARBUNYA', 'bombay': 'BOMBAY', 'cali': 'CALI',
            'dermason': 'DERMASON', 'D3RMAS0N': 'DERMASON',
            'horoz': 'HOROZ', 'H0R0Z': 'HOROZ',
            'seker': 'SEKER', 'S3K3R': 'SEKER',
            'sira': 'SIRA', 'B0MBAY': 'BOMBAY',
        }

    def load_raw(self, split='train'):
        """加载原始CSV"""
        path = f'{self.raw_dir}/Dry_Bean_Dataset_Dirty_{split}.csv'
        df = pd.read_csv(path)
        print(f"  原始{split}: {len(df)}条 × {len(df.columns)-1}特征")
        return df

    def clean_labels(self, df):
        """清洗类别标签"""
        df['Class'] = df['Class'].str.strip().replace(self.clean_map)
        return df

    def clean_features(self, df):
        """清洗特征值"""
        for c in ['Solidity', 'Compactness']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def fill_missing(self, train_df, val_df=None, test_df=None):
        """用中位数填充缺失值"""
        all_data = train_df[[c for c in train_df.columns if c != 'Class']]
        if val_df is not None:
            all_data = pd.concat([all_data, val_df[[c for c in val_df.columns if c != 'Class']]])
        if test_df is not None:
            all_data = pd.concat([all_data, test_df[[c for c in test_df.columns if c != 'Class']]])
        for col in all_data.columns:
            if all_data[col].isna().sum() > 0:
                med = all_data[col].median()
                train_df[col] = train_df[col].fillna(med)
                if val_df is not None: val_df[col] = val_df[col].fillna(med)
                if test_df is not None: test_df[col] = test_df[col].fillna(med)
        return train_df, val_df, test_df

    def scale_features(self, train_df, val_df=None, test_df=None, fit=True):
        """特征标准化"""
        self.feat_cols = [c for c in train_df.columns if c != 'Class']
        if fit:
            self.scaler = StandardScaler()
            train_df[self.feat_cols] = self.scaler.fit_transform(train_df[self.feat_cols])
        else:
            train_df[self.feat_cols] = self.scaler.transform(train_df[self.feat_cols])
        if val_df is not None:
            val_df[self.feat_cols] = self.scaler.transform(val_df[self.feat_cols])
        if test_df is not None:
            test_df[self.feat_cols] = self.scaler.transform(test_df[self.feat_cols])
        return train_df, val_df, test_df

    def load_data(self, split='train', cleaned=True):
        """加载（清洗后）数据，返回X, y, feature_names, class_names"""
        if cleaned:
            path = f'{self.clean_dir}/{split}_clean.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                feat_cols = [c for c in df.columns if c != 'Class']
                X = df[feat_cols].values
                y = df['Class'].values
                return X, y, feat_cols, None
        df = self.load_raw(split)
        df = self.clean_labels(df)
        df = self.clean_features(df)
        return df

    def save_clean(self, train_df, val_df, test_df):
        """保存清洗后数据"""
        train_df.to_csv(f'{self.clean_dir}/train_clean.csv', index=False)
        val_df.to_csv(f'{self.clean_dir}/val_clean.csv', index=False)
        test_df.to_csv(f'{self.clean_dir}/test_clean.csv', index=False)
        joblib.dump(self.scaler, 'models/drybean_scaler.pkl')
        print(f"  已保存清洗后数据到 {self.clean_dir}/")
        print(f"  已保存Scaler到 models/drybean_scaler.pkl")

    def get_X_y(self, split='train', cleaned=True):
        """获取可直接用于训练的numpy数组"""
        X, y, feat, _ = self.load_data(split, cleaned)
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
        else:
            y_enc = self.label_encoder.transform(y)
        return X.astype(np.float32), y_enc, feat, self.label_encoder.classes_


class TitanicLoader:
    """Titanic数据集加载器（预留）"""
    def load(self):
        print("Titanic loader - 待实现")
        return None, None, None, None


class MNISTLoader:
    """MNIST数据集加载器（预留）"""
    def load(self):
        print("MNIST loader - 待实现")
        return None, None, None, None


def get_loader(dataset='drybean'):
    """工厂方法：根据数据集名返回对应的加载器"""
    loaders = {
        'drybean': DryBeanLoader(),
        'titanic': TitanicLoader(),
        'mnist': MNISTLoader(),
    }
    return loaders.get(dataset, DryBeanLoader())
