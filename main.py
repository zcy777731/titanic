"""
Machine Learning Project - Main Entry
期末作业: Dry Bean Dataset 全流程
Usage:
  python main.py --algo=drybean --data=drybean --process=analyze
  python main.py --algo=drybean --data=drybean --process=preprocess
  python main.py --algo=drybean --data=drybean --process=experiments
  python main.py --algo=lr --data=drybean --process=train
  python main.py --algo=svm --data=drybean --process=train
  python main.py --algo=knn --data=drybean --process=train
  python main.py --algo=xgb --data=drybean --process=train
  python main.py --algo=all --data=all --process=all
"""
import subprocess, sys, os, argparse

def run_script(script_name, description, cwd=None):
    print(f">>> {description}...")
    src = os.path.join(os.path.dirname(__file__), "src")
    path = os.path.join(src, script_name)
    if cwd is None:
        cwd = os.path.dirname(__file__)
    result = subprocess.run([sys.executable, path], cwd=cwd)
    if result.returncode == 0:
        print(f"  [DONE] {description}")
    return result.returncode

def run_drybean(process):
    scripts = {
        'analyze': ('drybean_analysis.py', 'Data Analysis'),
        'preprocess': ('drybean_preprocessing.py', 'Data Preprocessing'),
        'experiments': ('drybean_experiments.py', 'Experiments'),
        'train': ('drybean_experiments.py', 'All Experiments'),
    }
    if process in scripts:
        run_script(*scripts[process])
    elif process == 'all':
        for script, desc in scripts.values():
            run_script(script, desc)

def run_drybean_single(algo):
    import pandas as pd, time
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    train = pd.read_csv('DryBeanDataset/train_clean.csv')
    test = pd.read_csv('DryBeanDataset/test_clean.csv')
    feat = [c for c in train.columns if c != 'Class']
    le = LabelEncoder()
    X_tr, y_tr = train[feat].values, le.fit_transform(train['Class'])
    X_te, y_te = test[feat].values, le.transform(test['Class'])

    if algo == 'lr':
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=1000, random_state=42)
    elif algo == 'svm':
        from sklearn.svm import SVC
        m = SVC(kernel='rbf', gamma='scale', random_state=42)
    elif algo == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        m = KNeighborsClassifier(n_neighbors=5)
    elif algo == 'xgb':
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=200, random_state=42, verbosity=0)
    else:
        print(f"Unknown algo: {algo}"); return

    t0 = time.time()
    m.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    print(f"{algo}: test acc={acc*100:.2f}%  train_t={time.time()-t0:.1f}s")
    import joblib; joblib.dump(m, f'models/drybean_{algo}.pkl')

def main():
    parser = argparse.ArgumentParser(description="ML Project")
    parser.add_argument("--algo", type=str, default="all")
    parser.add_argument("--data", type=str, default="all")
    parser.add_argument("--process", type=str, default="all")
    args = parser.parse_args()

    if args.data == 'drybean' and args.algo == 'drybean':
        run_drybean(args.process)
    elif args.data == 'drybean' and args.algo in ['lr','svm','knn','xgb']:
        run_drybean('preprocess')  # ensure data is clean
        run_drybean_single(args.algo)
    else:
        print("Use: python main.py --algo=drybean --data=drybean --process=analyze/preprocess/experiments")
        print("Or:  python main.py --algo=lr/svm/knn/xgb --data=drybean --process=train")

if __name__ == '__main__':
    main()
