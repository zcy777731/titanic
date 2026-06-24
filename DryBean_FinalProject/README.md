# Dry Bean Dataset — 期末大作业

机器学习与项目实践 2026_AIT209  
**作者**: 赵晨妤  
**GitHub**: https://github.com/zcy777731/titanic

---

## 项目简介

基于 Dry Bean Dataset（干豆数据集）的全流程机器学习分类项目。  
7种干豆 × 16个形态特征 × 13611条样本。

## 快速开始

```bash
# 全部实验
python main.py --process=all

# 数据分析
python main.py --process=analyze

# 数据预处理
python main.py --process=preprocess

# 训练单个算法
python main.py --algo=lr --process=train
python main.py --algo=svm --process=train
python main.py --algo=knn --process=train
python main.py --algo=xgb --process=train

# 全部算法对比实验
python main.py --process=experiments

# 额外加分维度
python main.py --process=extra
```

## 实验结果

| 算法 | 测试准确率 | 训练时间 | 推理时间 | 过拟合度 |
|------|-----------|---------|---------|---------|
| Logistic Regression | 91.85% | 0.17s | 6ms | 0.65% |
| SVM (RBF) | **92.88%** | 5.24s | 4526ms | **0.11%** |
| KNN (k=5) | 91.74% | **0.04s** | 785ms | 2.03% |
| XGBoost ⭐ | 92.58% | 4.31s | 116ms | 7.00% |

⭐ XGBoost = 课上未讲过的算法（加分项）

## 工程结构

```
├── main.py                      统一命令行入口
├── src/
│   ├── data_loader.py           数据加载+清洗
│   ├── trainer.py               模型训练+保存
│   ├── evaluator.py             测试评估+报告
│   ├── drybean_analysis.py      数据分析
│   ├── drybean_preprocessing.py 数据预处理
│   ├── drybean_experiments.py   多算法实验
│   ├── extra_comparisons.py     额外对比维度
│   └── beautify_charts.py       图表美化
├── DryBeanDataset/              清洗后数据
├── models/                      训练好的模型
├── results/                     实验报告
└── tmp_imgs/                    19张可视化图表
```

## 依赖

```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib
```
