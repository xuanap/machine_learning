# A 的 K-means / GMM 接口与实验使用说明

A 的两种算法、参数/消融框架、客户画像和分群比较工具已经实现。仓库已有 C 的谱聚类预处理代码，但尚未导出统一预处理矩阵及元数据，五算法正式比较协议也待共同确认。原始尺度 smoke 结果只验证代码，不作为正式算法效果。

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

## 4. C 提供什么后可以启动正式实验

1. 处理后的 CSV：CustomerID 与原六特征列，客户集合和源表相同；行顺序可不同，但每行必须保留正确 ID。
2. 预处理说明 JSON：原特征表及输出文件的 SHA-256、数据版本、特征列顺序、处理版本、实际 scaler 名称、实际 log1p 列列表。
3. C 保存拟合后的处理器或均值、尺度等参数，以便复现。不要把仅用于画图的 PCA 二维坐标替代六特征基线输入。

模板：config/preprocessing_metadata.example.json。里面有占位符，**不是已经完成的预处理结果**，不能直接用作正式元数据。填写实际信息后另存文件。

SHA-256 可这样计算：

```python
import hashlib
from pathlib import Path
digest = hashlib.sha256(Path("data/customer_features_scaled.csv").read_bytes()).hexdigest()
```

命令示例：

```text
.venv\Scripts\python main.py --mode formal --suite baseline --input data/customer_features_scaled.csv --metadata data/preprocessing_metadata.json
.venv\Scripts\python main.py --mode formal --suite parameters --input data/customer_features_scaled.csv --metadata data/preprocessing_metadata.json
.venv\Scripts\python main.py --mode formal --suite ablation --input data/customer_features_scaled.csv --metadata data/preprocessing_metadata.json
```

默认正式配置为 config/experiment_config.json：K=2..8、三个随机种子、两种 K-means 初始化、四种 GMM 协方差类型，特征消融先安排 RFM 与六特征。它们是待对齐的起步方案，不是已完成的实验结果。

命令行入口会检查元数据哈希、客户覆盖、缺失值、重复 ID 和表结构；拒绝缺少预处理说明的正式运行和未改变的原始 CSV。元数据只能记录 C 声明的处理过程，不能仅靠 JSON 自动证明标准化算法实现正确。

当前入口支持逐特征的 StandardScaler、RobustScaler、MinMaxScaler 及 log1p。消融通过选择已处理矩阵的相应列进行；若小组采用需要跨特征共同拟合的其他变换，应先调整这部分协议。

## 5. 输出和评价口径

每个运行保存：labels.csv、centers.csv、cluster_sizes.csv、run.json；GMM 另存 probabilities.csv。顶层保存 summary.csv、experiment_manifest.json、sample_customer_ids.csv、NOTICE.txt 和图表。

即使配置运行失败，也会在 summary.csv 和 run.json 留下参数与错误；未收敛、退化成更少簇等情况不伪装成正常结果。已有输出目录不会自动覆盖。

本地参考指标位于 experiments/reference_metrics.py：Silhouette、DB、CH；Silhouette 最多取固定种子的 1,000 个样本，DB 和 CH 使用全体输入。它是 A 用来验证框架的实现，**尚未替代 C 的统一指标**。不同种子之间的正式汇总和稳定性仍按 C 的协议进行。

C 可以只导入算法，再统一计算指标。也可在 Python 调用 experiments.runner.run_suite 时通过 metric_fn 和 metrics_provider 接入 C 的评价函数。预处理数据齐备后，指标协议仍需对齐，不能直接把标有 A_reference_pending_C 的结果宣称为五算法最终对比。

特征消融的内部指标处于不同特征空间，不能仅因数值上升便认定新增特征更好；应结合客户画像和分群一致性分析。

## 6. 下一步

当前可把两种算法和本说明交给 C 对接。取得统一处理结果及指标协议后，运行全客户正式基线，再做参数实验及与 B 约定的特征消融，最后才整理客户画像与报告结论。

结果分析工具见 docs/customer_analysis.md，两算法比较见 docs/cluster_comparison.md。分支接入范围和待对齐事项见 docs/a_handoff.md。
