"""
Dry Bean Dataset — 期末大作业
机器学习与项目实践 2026_AIT209
================================
基于 Dry Bean Dataset 的全流程机器学习工程项目。
涵盖：数据分析 → 数据清洗 → 特征工程 → 多算法实验 → 系统集成。

Usage:
  # 全流程
  python main.py --process=all

  # 数据分析
  python main.py --process=analyze

  # 数据预处理
  python main.py --process=preprocess

  # 训练模型
  python main.py --algo=lr --process=train
  python main.py --algo=svm --process=train
  python main.py --algo=knn --process=train
  python main.py --algo=xgb --process=train

  # 算法对照实验
  python main.py --process=experiments

  # 额外对比维度（加分项）
  python main.py --process=extra

  # 列出支持算法
  python main.py --algo=list --process=train
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')


def banner():
    print("=" * 60)
    print("  Dry Bean Dataset — 期末大作业")
    print("  机器学习与项目实践 2026_AIT209")
    print("=" * 60)


def run_analysis():
    """Part 1: 数据分析"""
    banner()
    print("\n>>> 模块: 数据分析")
    print(">>> 功能: 数据描述 / 污染观察 / 可视化\n")
    exec(open('src/drybean_analysis.py').read())


def run_preprocess():
    """Part 2: 数据预处理"""
    banner()
    print("\n>>> 模块: 数据预处理")
    print(">>> 功能: 标签清洗 / 特征清洗 / 缺失值填充 / 标准化\n")
    from data_loader import DryBeanLoader
    loader = DryBeanLoader(clean_dir='DryBeanDataset')

    print("  加载原始数据...")
    train = loader.load_raw('train')
    val = loader.load_raw('val')
    test = loader.load_raw('test')

    print("  清洗类别标签 + 特征值...")
    for df in [train, val, test]:
        loader.clean_labels(df)
        loader.clean_features(df)

    print("  填充缺失值...")
    train, val, test = loader.fill_missing(train, val, test)

    print("  特征标准化...")
    train, val, test = loader.scale_features(train, val, test, fit=True)

    print("  保存清洗后数据...")
    loader.save_clean(train, val, test)

    print("\n  ✅ 预处理完成")
    print(f"  训练: {len(train)} | 验证: {len(val)} | 测试: {len(test)}")


def run_train(algo):
    """训练指定算法"""
    from data_loader import DryBeanLoader
    from trainer import train_model, save_model, list_algorithms, ALGORITHMS

    if algo == 'list':
        list_algorithms()
        return

    banner()
    print(f"\n>>> 模块: 模型训练")
    print(f">>> 算法: {ALGORITHMS[algo]['name']}\n")

    loader = DryBeanLoader(clean_dir='DryBeanDataset')
    X, y, feat, classes = loader.get_X_y('train', cleaned=True)
    print(f"  训练样本: {len(X):,}  特征: {len(feat)}  类别: {len(classes)}")
    print(f"  算法参数: {ALGORITHMS[algo]['params']}")

    model, train_t = train_model(X, y, algo)
    path = save_model(model, algo, 'models')

    print(f"\n  ✅ 训练完成")
    print(f"  耗时: {train_t:.2f}s")
    print(f"  模型: {path}")


def run_experiments():
    """全部实验: 4种算法对比"""
    from trainer import train_model, save_model, ALGORITHMS
    from data_loader import DryBeanLoader
    from evaluator import Evaluator, ResultCollector

    banner()
    print("\n>>> 模块: 多算法实验对比")
    print(">>> 算法: LR / SVM / KNN / XGBoost（含课上未讲过算法）\n")

    loader = DryBeanLoader(clean_dir='DryBeanDataset')
    X_tr, y_tr, feat, c = loader.get_X_y('train', cleaned=True)
    X_te, y_te, _, _ = loader.get_X_y('test', cleaned=True)

    print(f"  训练集: {len(X_tr):,} 条")
    print(f"  测试集: {len(X_te):,} 条")
    print(f"  特征数: {len(feat)} | 类别数: {len(c)}")
    print()

    collector = ResultCollector()
    results = []

    for algo in ['lr', 'svm', 'knn', 'xgb']:
        name = ALGORITHMS[algo]['name']
        stars = " ⭐" if algo == 'xgb' else ""
        print(f"─── {name}{stars} ───")

        model, t = train_model(X_tr, y_tr, algo)
        save_model(model, algo, 'models')
        ev = Evaluator(model, algo, X_tr, y_tr, X_te, y_te)
        ev.evaluate()
        ev.robustness_test()
        ev.summary()
        ev.results.update({
            'n_train': len(X_tr), 'n_test': len(X_te),
            'n_features': len(feat), 'n_classes': len(c),
        })
        collector.add(algo, model, ev.results)
        results.append((algo, name, ev.results))
        print()

    # 汇总表
    print("\n" + "=" * 70)
    print("结果汇总表")
    print("=" * 70)
    df = collector.to_dataframe()
    print(df.to_string(index=False))
    print()

    # 最佳
    best = max(collector.records, key=lambda r: r.get('test_acc', 0))
    fastest = min(collector.records, key=lambda r: r.get('infer_time_ms', 9999))
    most_robust = max(collector.records, key=lambda r: r.get('robustness', {}).get(1.0, 0))
    least_overfit = min(collector.records, key=lambda r: r.get('overfitting', 1))

    print(f"🏆 最佳准确率:   {best['algo_key']} ({best.get('test_acc', 0)*100:.2f}%)")
    print(f"⚡ 最快推理:     {fastest['algo_key']} ({fastest.get('infer_time_ms', 0):.1f}ms)")
    print(f"🛡️ 最鲁棒:       {most_robust['algo_key']} (σ=1.0时{most_robust.get('robustness',{}).get(1.0,0)*100:.1f}%)")
    print(f"✅ 过拟合最小:   {least_overfit['algo_key']} ({least_overfit.get('overfitting',0)*100:.2f}%)")
    print(f"⭐ XGBoost: 课上未讲过的算法（加分项）")

    # 保存
    collector.save_report('results/drybean_accuracy_report.txt')


def run_extra():
    """额外对比维度（加分项）"""
    banner()
    print("\n>>> 模块: 额外对比维度（加分项）")
    print(">>> ① 模型大小 vs 参数量")
    print(">>> ② 数据量敏感性曲线")
    print(">>> ③ 训练时间分解")
    print(">>> ④ 可解释性对比\n")
    exec(open('src/extra_comparisons.py').read())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Dry Bean Dataset — 期末大作业',
        epilog="示例:\n"
               "  python main.py --process=all\n"
               "  python main.py --algo=lr --process=train\n"
               "  python main.py --process=experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--algo', type=str, default='lr',
                        help='算法: lr / svm / knn / xgb / list')
    parser.add_argument('--process', type=str, default='all',
                        help='流程: analyze / preprocess / train / experiments / extra / all')
    args = parser.parse_args()

    start_t = time.time()

    if args.process == 'analyze':
        run_analysis()
    elif args.process == 'preprocess':
        run_preprocess()
    elif args.process == 'train':
        run_train(args.algo)
    elif args.process == 'experiments':
        run_experiments()
    elif args.process == 'extra':
        run_extra()
    elif args.process == 'all':
        run_analysis()
        run_preprocess()
        run_experiments()
        print(f"\n{'='*60}")
        print("  全流程完成 ✅")
        print(f"  总耗时: {time.time()-start_t:.1f}s")
        print(f"{'='*60}")
    else:
        print(f"未知流程: {args.process}")
        print("可用: analyze / preprocess / train / experiments / extra / all")


if __name__ == '__main__':
    main()
