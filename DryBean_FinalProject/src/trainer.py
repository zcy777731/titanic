"""
模型训练模块
支持: Logistic Regression / SVM / KNN / XGBoost
功能: 训练 / 保存模型 / 加载模型
"""
import sys, time, numpy as np, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ALGORITHMS = {
    'lr': {
        'name': 'Logistic Regression',
        'class': None,
        'params': {'max_iter': 1000, 'C': 1.0, 'solver': 'lbfgs'},
        'desc': '线性分类器基准，L2正则化，多分类'
    },
    'svm': {
        'name': 'SVM (RBF)',
        'class': None,
        'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        'desc': '非线性分类器，RBF核技巧，间隔最大化'
    },
    'knn': {
        'name': 'KNN (k=5)',
        'class': None,
        'params': {'n_neighbors': 5, 'algorithm': 'kd_tree'},
        'desc': '非参数模型，基于k近邻投票'
    },
    'xgb': {
        'name': 'XGBoost',
        'class': None,
        'params': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
        'desc': '梯度提升树，课上未讲过的算法（加分项）'
    },
}


def get_model(algo_key):
    """根据算法名返回模型实例"""
    if algo_key not in ALGORITHMS:
        raise ValueError(f"不支持的算法: {algo_key}，可选: {list(ALGORITHMS.keys())}")

    info = ALGORITHMS[algo_key]
    if algo_key == 'lr':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**info['params'], random_state=42)
    elif algo_key == 'svm':
        from sklearn.svm import SVC
        return SVC(**info['params'], probability=True, random_state=42)
    elif algo_key == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**info['params'])
    elif algo_key == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(**info['params'], random_state=42, verbosity=0)


def train_model(X_train, y_train, algo='lr'):
    """训练单个模型"""
    print(f"  训练 {ALGORITHMS[algo]['name']}...")
    model = get_model(algo)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_t = time.time() - t0
    print(f"    完成 ({train_t:.2f}s)")
    return model, train_t


def save_model(model, algo, path='models'):
    """保存模型"""
    import joblib, os
    os.makedirs(path, exist_ok=True)
    p = f'{path}/drybean_{algo}.pkl'
    joblib.dump(model, p)
    print(f"  模型已保存: {p}")
    return p


def load_model(algo, path='models'):
    """加载模型"""
    import joblib
    p = f'{path}/drybean_{algo}.pkl'
    model = joblib.load(p)
    print(f"  模型已加载: {p}")
    return model


def list_algorithms():
    """列出所有支持的算法"""
    print(f"\n支持的算法 ({len(ALGORITHMS)}种):")
    for key, info in ALGORITHMS.items():
        print(f"  {key:5s} | {info['name']:22s} | {info['desc']}")
    print(f"\n  其中 'xgb' 为课上未讲过的算法（⭐ 加分项）")
