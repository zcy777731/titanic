"""
Dry Bean Dataset — 期末大作业
机器学习与项目实践 2026_AIT209

Usage:
  python main.py --process=analyze       数据分析
  python main.py --process=preprocess    数据预处理
  python main.py --algo=lr --process=train  训练LR
  python main.py --algo=svm --process=train 训练SVM
  python main.py --algo=knn --process=train 训练KNN
  python main.py --algo=xgb --process=train 训练XGBoost
  python main.py --process=experiments   全部实验
  python main.py --process=extra         额外对比维度
  python main.py --process=all           全流程
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')


def run_analysis():
    exec(open('src/drybean_analysis.py').read())

def run_preprocess():
    from data_loader import DryBeanLoader
    loader = DryBeanLoader(clean_dir='DryBeanDataset')
    train = loader.load_raw('train')
    val = loader.load_raw('val')
    test = loader.load_raw('test')
    for df in [train, val, test]:
        loader.clean_labels(df); loader.clean_features(df)
    train, val, test = loader.fill_missing(train, val, test)
    train, val, test = loader.scale_features(train, val, test, fit=True)
    loader.save_clean(train, val, test)
    print("预处理完成 ✅")

def run_train(algo):
    from data_loader import DryBeanLoader
    from trainer import train_model, save_model, list_algorithms
    if algo == 'list':
        list_algorithms(); return
    loader = DryBeanLoader(clean_dir='DryBeanDataset')
    X, y, feat, classes = loader.get_X_y('train', cleaned=True)
    print(f"训练: {len(X)}条  特征: {len(feat)}  类别: {len(classes)}")
    model, t = train_model(X, y, algo)
    save_model(model, algo, 'models')
    print(f"完成 ({t:.2f}s)")

def run_experiments():
    from trainer import train_model, save_model, ALGORITHMS
    from data_loader import DryBeanLoader
    from evaluator import Evaluator, ResultCollector
    loader = DryBeanLoader(clean_dir='DryBeanDataset')
    X_tr, y_tr, feat, c = loader.get_X_y('train', cleaned=True)
    X_te, y_te, _, _ = loader.get_X_y('test', cleaned=True)
    print(f"训练: {len(X_tr)}  测试: {len(X_te)}  特征: {len(feat)}")
    collector = ResultCollector()
    for algo in ['lr', 'svm', 'knn', 'xgb']:
        print(f"\n--- {ALGORITHMS[algo]['name']} ---")
        model, t = train_model(X_tr, y_tr, algo)
        save_model(model, algo, 'models')
        ev = Evaluator(model, algo, X_tr, y_tr, X_te, y_te)
        ev.evaluate()
        ev.robustness_test()
        ev.summary()
        ev.results.update({'n_train': len(X_tr), 'n_test': len(X_te),
                          'n_features': len(feat), 'n_classes': len(c)})
        collector.add(algo, model, ev.results)
    collector.save_report('results/drybean_accuracy_report.txt')
    print(collector.to_dataframe().to_string(index=False))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dry Bean Final Project')
    parser.add_argument('--algo', default='lr')
    parser.add_argument('--process', default='all')
    args = parser.parse_args()
    print(f"Dry Bean Final Project | process={args.process}")

    if args.process == 'analyze': run_analysis()
    elif args.process == 'preprocess': run_preprocess()
    elif args.process == 'train': run_train(args.algo)
    elif args.process == 'experiments': run_experiments()
    elif args.process == 'extra':
        exec(open('src/extra_comparisons.py').read())
    elif args.process == 'all':
        run_analysis(); run_preprocess(); run_experiments()
        print("\n全流程完成 ✅")

if __name__ == '__main__':
    main()
