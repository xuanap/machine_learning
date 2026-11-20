# A 的 K-means / GMM 接口与实验使用说明

A 的两种算法、参数/消融框架、客户画像和分群比较工具已经实现。已接入 C 的共享预处理函数，不要求 C 另导出 CSV；五算法最终整合与计时协议仍由小组确认。原始尺度 smoke 结果只验证代码，不作为正式算法效果。

下面是 A 提出的接入约定，可由 C 采用或在对齐后调整；不代表小组已经确认。当前五种算法的最终统一入口仍由 C 负责。

## 1. C 可以直接调用两种算法

```python
from algorithms.kmeans import fit_predict as run_kmeans
from algorithms.gmm import fit_predict as run_gmm

# X 是 C 统一预处理后的二维数值矩阵；不含 CustomerID。
km = run_kmeans(X, params={"n_clusters": 4, "n_init": 10}, random_state=42, threads=1)
gm = run_gmm(X, params={"n_components": 4, "covariance_type": "full", "n_init": 3},
             random_state=42, threads=1)

labels = km.labels
gmm_labels = gm.labels
memberships = gm.probabilities
```

共同签名：`fit_predict(X, params=None, random_state=42, threads=1)`。

两种方法都不会做 log1p、标准化或 PCA。可直接接 C 的矩阵，不必采用 A 的命令行实验入口。固定随机种子、初始化次数等参数；不在 params 中另传 random_state。

返回 `ClusteringResult`：

| 属性 | 含义 |
| --- | --- |
| algorithm | kmeans 或 gmm |
| labels | 长度等于样本数，顺序严格对应输入 X |
| centers | K-means 中心 / GMM 均值，位于输入特征空间 |
| probabilities | GMM 为 n_samples × n_components；K-means 为 None |
| fit_seconds | 仅 estimator.fit 耗时，包括其内部初始化 |
| params、seed | 实际模型参数和随机种子 |
| status | ok、not_converged、iteration_limit_reached 或 collapsed_clusters |
| n_iter、converged、warnings | 迭代数、可获得的收敛标志和警告 |
| diagnostics | K-means inertia；GMM AIC/BIC、权重等，以及线程数 |

计时不含外部预处理、预测、评价、读写和画图。默认限制数值库线程数为 1，便于在同一协议下比较。K-means 没有 sklearn 的 converged_ 属性，因此 converged 返回 None；达到迭代上限时明确标记，不伪造收敛判断。

GMM 的 AIC/BIC 只用于同一个数据和特征表示下的 GMM 配置比较，不与 K-means 或不同维数的特征实验横比。

## 2. 安装和验证

建议 Python 3.12，requirements-algorithms.txt 引用共享 requirements.txt；统一采用仓库已有的 scikit-learn / SciPy 版本，不引入 A 原本本地环境的另一套版本。虚拟环境不提交。

新环境可运行：

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-algorithms.txt
.venv\Scripts\python -m unittest discover -s tests -v
```

## 3. 现在可以运行的代码验证

```text
.venv\Scripts\python main.py --mode smoke --suite all
```

默认固定抽取 300 位客户，使用原始尺度六特征，只验证程序是否跑通。可通过 --sample-size 修改诊断样本数。输出位于 results/smoke/ 下，每次新建目录。不要把这里的指标用于最佳参数选择或小组最终结论。

仓库保留 `results/smoke/a_framework_check_v2/` 作为已有回归测试夹具；它记录先前 300 位客户的 16 次试跑，并非本分支的重新运行结果。用上述命令可运行当前代码，输出会另建目录。小规模配置见 config/smoke_experiments.json；图上带 SMOKE ONLY 标识。当前绘图保存 plot_support.csv，记录有效/计划运行数；只有一次有效运行时标准差缺失，不画成零误差线。

## 4. 直接使用 C 的共享函数运行全客户实验

默认正式入口读取现有 data/customer_features.csv，调用 preprocessing.feature_scaling.load_customer_features 一次，使用返回的原表顺序 CustomerID 和六维矩阵。不会重复标准化，不使用 PCA 二维绘图坐标。

```text
python main.py --mode formal --suite baseline
python main.py --mode formal --suite parameters
python main.py --mode formal --suite ablation
```

源表及共享代码 SHA-256、log1p 列、特征顺序、实际矩阵哈希、环境版本均自动记录。每个实验目录保存 model_input.csv 作为精确可复现的拟合坐标快照；这由 A 的运行程序自动完成，不要求 C 另供数据或手填元数据。现有 CSV 保留原始尺度，用于画像。

旧的外部处理后 CSV + JSON 接入方式仍支持：显式同时提供 --input 和 --metadata；这是可选兼容入口，不是当前实验的前置条件。仅指定一个参数会报错。

本轮配置为 config/experiment_config.json：K=2..8、种子 0/42/2026、两种 K-means 初始化、四种 GMM 协方差类型。RFM 与六特征对照是 A 先行运行的方案，最终消融分工仍与 B 确认。

## 5. 输出和评价口径

每个运行保存：labels.csv、centers.csv、cluster_sizes.csv、run.json；GMM 另存 probabilities.csv。顶层保存 summary.csv、experiment_manifest.json、sample_customer_ids.csv、model_input.csv、NOTICE.txt 和图表。

即使配置运行失败，也会在 summary.csv 和 run.json 留下参数与错误；未收敛、退化成更少簇等情况不伪装成正常结果。已有输出目录不会自动覆盖。

原始尺度 smoke 仍使用 experiments/reference_metrics.py 的参考指标。正式命令使用 experiments/group_metrics.py：在实际拟合特征空间对全部 4,338 位客户计算 Euclidean Silhouette、DB 和 CH，定义与 C 当前 run_spectral.py 一致；记录名为 full_euclidean_matches_C_spectral_current。这表示与当前代码对齐，不冒称五算法最终协议已获全组批准。

A 的算法拟合和指标计算均限制数值库线程为 1；计时只包含 fit。C 的谱聚类入口尚未采用相同线程限制，五算法最终耗时横比仍需统一运行环境与范围。C 可直接导入算法，也可通过 run_suite 的 metric_fn / metrics_provider 接入未来的统一评价实现。

特征消融的内部指标处于不同特征空间，不能仅因数值上升便认定新增特征更好；应结合客户画像和分群一致性分析。

## 6. 下一步

已完成 6 次基线、60 次参数实验及 12 次 A 的特征对照，均使用 C 的共享预处理。结果和边界见 docs/a_experiment_results.md；这些是 A 的真实全客户实验，尚不构成五算法最终结论。

结果分析工具见 docs/customer_analysis.md，两算法比较见 docs/cluster_comparison.md。分支接入范围和待对齐事项见 docs/a_handoff.md。
