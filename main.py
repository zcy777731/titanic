"""
Machine Learning Project — 统一命令行接口
期末作业: Dry Bean Dataset 全流程

Usage:
  # 数据分析
  python main.py --data=drybean --process=analyze

  # 数据预处理
  python main.py --data=drybean --process=preprocess

  # 训练单个算法
  python main.py --data=drybean --algo=lr --process=train
  python main.py --data=drybean --algo=svm --process=train
  python main.py --data=drybean --algo=knn --process=train
  python main.py --data=drybean --algo=xgb --process=train

  # 测试评估
  python main.py --data=drybean --algo=lr --process=test
  python main.py --data=drybean --algo=xgb --process=test

  # 全部实验（训练+评估+对比）
  python main.py --data=drybean --process=experiments

  # 列出支持的算法
  python main.py --data=drybean --process=list

  # 全流程（分析→预处理→实验）
  python main.py --data=drybean --process=all
"""
import sys, os, argparse, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)


def run_analysis():
    """数据分析"""
    print("\n>>> 数据分析...")
    exec(open(os.path.join(SRC_DIR, 'drybean_analysis.py')).read())


def run_preprocess():
    """数据预处理"""
    print("\n>>> 数据预处理...")
    from data_loader import DryBeanLoader
    loader = DryBeanLoader()
    train = loader.load_raw('train')
    val = loader.load_raw('val')
    test = loader.load_raw('test')
    for df in [train, val, test]:
        loader.clean_labels(df)
        loader.clean_features(df)
    train, val, test = loader.fill_missing(train, val, test)
    train, val, test = loader.scale_features(train, val, test, fit=True)
    loader.save_clean(train, val, test)
    print("  预处理完成 ✅")


def run_train(algo):
    """训练指定算法"""
    from data_loader import DryBeanLoader
    from trainer import train_model, save_model, list_algorithms
    if algo == 'list':
        list_algorithms()
        return
    loader = DryBeanLoader()
    X, y, feat, classes = loader.get_X_y('train', cleaned=True)
    print(f"\n>>> 训练 {algo}...")
    print(f"  训练样本: {len(X)}  特征: {len(feat)}  类别: {len(classes)}")
    model, train_t = train_model(X, y, algo)
    save_model(model, algo)
    print(f"  训练完成 ✅ ({train_t:.2f}s)")


def run_test(algo):
    """测试评估指定算法"""
    from data_loader import DryBeanLoader
    from trainer import load_model, ALGORITHMS
    from evaluator import Evaluator
    loader = DryBeanLoader()
    X_tr, y_tr, feat, classes = loader.get_X_y('train', cleaned=True)
    X_te, y_te, _, _ = loader.get_X_y('test', cleaned=True)
    model = load_model(algo)
    print(f"\n>>> 评估 {algo}...")
    print(f"  测试样本: {len(X_te)}")
    ev = Evaluator(model, algo, X_tr, y_tr, X_te, y_te)
    ev.evaluate()
    ev.robustness_test()
    ev.summary()


def run_experiments():
    """全部实验（训练4种算法 + 评估 + 对比）"""
    from data_loader import DryBeanLoader
    from trainer import train_model, save_model, ALGORITHMS
    from evaluator import Evaluator, ResultCollector
    import numpy as np

    print("\n" + "="*60)
    print("Dry Bean Dataset — 全实验")
    print("="*60)

    loader = DryBeanLoader()
    X_tr, y_tr, feat, classes = loader.get_X_y('train', cleaned=True)
    X_te, y_te, _, _ = loader.get_X_y('test', cleaned=True)
    print(f"训练: {len(X_tr)}  测试: {len(X_te)}  特征: {len(feat)}  类别: {len(classes)}")

    collector = ResultCollector()
    models = {}

    for algo in ['lr', 'svm', 'knn', 'xgb']:
        print(f"\n--- {ALGORITHMS[algo]['name']} ---")
        model, train_t = train_model(X_tr, y_tr, algo)
        save_model(model, algo)
        ev = Evaluator(model, algo, X_tr, y_tr, X_te, y_te)
        ev.evaluate()
        ev.robustness_test()
        ev.summary()
        ev.results['n_train'] = len(X_tr)
        ev.results['n_test'] = len(X_te)
        ev.results['n_features'] = len(feat)
        ev.results['n_classes'] = len(classes)
        collector.add(algo, model, ev.results)
        models[algo] = model

    # 保存汇总报告
    report_path = collector.save_report()
    print(f"\n  汇总报告: {report_path}")

    # 打印汇总表
    df = collector.to_dataframe()
    print(f"\n{'='*70}")
    print("汇总表")
    print('='*70)
    print(df.to_string(index=False))

    # 最佳算法
    best = max(collector.records, key=lambda r: r.get('test_acc',0))
    print(f"\n🏆 最佳准确率: {best['algo_key']} ({best.get('test_acc',0)*100:.2f}%)")
    fastest = min(collector.records, key=lambda r: r.get('infer_time_ms',9999))
    print(f"⚡ 最快推理: {fastest['algo_key']} ({fastest.get('infer_time_ms',0):.1f}ms)")
    print(f"⭐ XGBoost为课上未讲过的算法（加分项）")


def main():
    parser = argparse.ArgumentParser(description='ML Project - Dry Bean Classification')
    parser.add_argument('--data', type=str, default='drybean', help='Dataset name')
    parser.add_argument('--algo', type=str, default='lr', help='Algorithm: lr/svm/knn/xgb/list')
    parser.add_argument('--process', type=str, default='all',
                        help='Process: analyze/preprocess/train/test/experiments/list/all')
    args = parser.parse_args()

    print(f"ML Project | data={args.data} | algo={args.algo} | process={args.process}")

    if args.process == 'analyze':
        run_analysis()
    elif args.process == 'preprocess':
        run_preprocess()
    elif args.process == 'train':
        run_train(args.algo)
    elif args.process == 'test':
        run_test(args.algo)
    elif args.process == 'experiments':
        run_experiments()
    elif args.process == 'list':
        from trainer import list_algorithms
        list_algorithms()
    elif args.process == 'all':
        run_analysis()
        run_preprocess()
        run_experiments()
        print(f"\n全流程完成 ✅")


if __name__ == '__main__':
    main()
