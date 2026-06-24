"""
模型评估模块
功能: 预测 / 准确率计算 / 鲁棒性测试 / 过拟合分析 / 结果报告
"""
import sys, time, numpy as np, pandas as pd, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Evaluator:
    """模型评估器"""

    def __init__(self, model, algo_key, X_train, y_train, X_test, y_test):
        self.model = model
        self.algo_key = algo_key
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

    def evaluate(self):
        """运行全部评估"""
        self.test_accuracy()
        self.speed_test()
        return self.results

    def test_accuracy(self):
        """测试准确率"""
        t0 = time.time()
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        self.infer_time = (time.time() - t0) / 2 * 1000  # ms per prediction

        self.results['train_acc'] = accuracy_score(self.y_train, y_pred_train)
        self.results['test_acc'] = accuracy_score(self.y_test, y_pred_test)
        self.results['overfitting'] = self.results['train_acc'] - self.results['test_acc']

    def speed_test(self):
        """推理速度测试"""
        t0 = time.time()
        for _ in range(10):
            self.model.predict(self.X_test[:100])
        self.results['infer_time_ms'] = (time.time() - t0) / 10

    def robustness_test(self, noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0]):
        """鲁棒性测试（添加高斯噪声）"""
        self.results['robustness'] = {}
        for noise in noise_levels:
            X_noisy = self.X_test + np.random.randn(*self.X_test.shape) * noise
            acc = accuracy_score(self.y_test, self.model.predict(X_noisy))
            self.results['robustness'][noise] = acc
        return self.results['robustness']

    def get_confusion_matrix(self):
        """计算混淆矩阵"""
        y_pred = self.model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)

    def summary(self):
        """打印评估摘要"""
        r = self.results
        print(f"\n{'='*50}")
        print(f"评估结果: {self.algo_key}")
        print('='*50)
        print(f"  训练准确率:  {r.get('train_acc',0)*100:.2f}%")
        print(f"  测试准确率:  {r.get('test_acc',0)*100:.2f}%")
        print(f"  过拟合度:    {r.get('overfitting',0)*100:.2f}%")
        print(f"  推理时间:    {r.get('infer_time_ms',0):.1f}ms")
        print(f"  {'鲁棒性(σ=0.5):':14s} {r.get('robustness',{}).get(0.5,0)*100:.1f}%")
        print(f"  {'鲁棒性(σ=1.0):':14s} {r.get('robustness',{}).get(1.0,0)*100:.1f}%")


class ResultCollector:
    """多算法结果汇总"""

    def __init__(self):
        self.records = []

    def add(self, algo_key, model, results):
        """添加一个算法的结果"""
        self.records.append({
            'algo_key': algo_key,
            'model': model,
            **results
        })

    def to_dataframe(self):
        """转为DataFrame"""
        rows = []
        for r in self.records:
            rows.append({
                'Algorithm': r['algo_key'],
                'Train_Acc': f"{r.get('train_acc',0)*100:.2f}%",
                'Test_Acc': f"{r.get('test_acc',0)*100:.2f}%",
                'Overfitting': f"{r.get('overfitting',0)*100:.2f}%",
                'Infer_Time': f"{r.get('infer_time_ms',0):.1f}ms",
                'Robust_0.5': f"{r.get('robustness',{}).get(0.5,0)*100:.1f}%",
                'Robust_1.0': f"{r.get('robustness',{}).get(1.0,0)*100:.1f}%",
            })
        return pd.DataFrame(rows)

    def save_report(self, path='results/drybean_accuracy_report.txt'):
        """保存结果报告"""
        df = self.to_dataframe()
        with open(path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Dry Bean Dataset — 多算法实验结果汇总\n")
            f.write("="*70 + "\n\n")
            f.write("实验环境:\n")
            f.write(f"  训练集: {self.records[0].get('n_train','?')}条\n")
            f.write(f"  测试集: {self.records[0].get('n_test','?')}条\n")
            f.write(f"  特征数: {self.records[0].get('n_features','?')}个\n")
            f.write(f"  类别数: {self.records[0].get('n_classes','?')}\n\n")
            f.write("准确率对比:\n")
            f.write(df[['Algorithm','Train_Acc','Test_Acc','Overfitting']].to_string(index=False))
            f.write("\n\n")
            f.write("推理速度:\n")
            f.write(df[['Algorithm','Infer_Time']].to_string(index=False))
            f.write("\n\n")
            f.write("鲁棒性测试:\n")
            f.write(df[['Algorithm','Robust_0.5','Robust_1.0']].to_string(index=False))
            f.write("\n\n")
            # 最佳标注
            best_test = max(self.records, key=lambda r: r.get('test_acc',0))
            f.write(f"最佳准确率: {best_test['algo_key']} ({best_test.get('test_acc',0)*100:.2f}%)\n")
            fastest = min(self.records, key=lambda r: r.get('infer_time_ms',9999))
            f.write(f"最快推理: {fastest['algo_key']} ({fastest.get('infer_time_ms',0):.1f}ms)\n")
            robust_best = max(self.records, key=lambda r: r.get('robustness',{}).get(1.0,0))
            f.write(f"最鲁棒: {robust_best['algo_key']} (σ=1.0时{robust_best.get('robustness',{}).get(1.0,0)*100:.1f}%)\n")
            min_overfit = min(self.records, key=lambda r: r.get('overfitting',1))
            f.write(f"过拟合最小: {min_overfit['algo_key']} ({min_overfit.get('overfitting',0)*100:.2f}%)\n")
        print(f"\n报告已保存: {path}")
        return path
