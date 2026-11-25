# 客户聚类五算法基准与综合分析

[汇报图 PNG](../results/comparison/shared_five_algorithms/benchmark.png) · [完整数据 CSV](../results/comparison/shared_five_algorithms/benchmark.csv) · [三算法原报告](shared_three_algorithms_archive.md)

五种算法均使用 Online Retail 的 4,338 位客户和同一组六维建模坐标。四个偏态特征经 log1p 后，六维特征统一标准化。三算法归档实际使用的矩阵与 B 正式归档的 float64 矩阵 SHA-256 一致，源特征表指纹也一致。评分表由归档汇总生成。同机计时仅复跑固定基准配置。

| 算法与配置 | 归档运行数 | 有效簇 | 评分人数 | 噪声 | 覆盖率 | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K-means / K=4，k-means++ | 3 | 4 | 4,338 | 0 | 100% | 0.2504 | 1.2877 | 2055.93 |
| GMM / 4 分量，全协方差 | 3 | 4 | 4,338 | 0 | 100% | 0.1060 | 2.3880 | 1255.27 |
| 谱聚类 / K=4，RBF γ=0.1 | 3 | 4 | 4,338 | 0 | 100% | 0.2559 | 1.3358 | 1980.64 |
| 层次聚类 / Ward，K=4 | 1 | 4 | 4,338 | 0 | 100% | 0.1882 | 1.5516 | 1717.27 |
| DBSCAN / ε=1.25，min_samples=5 | 1 | 2 | 4,306 | 32 | 99.26% | 0.5311 | 0.4393 | 33.39 |

评分均使用建模空间的欧氏距离；噪声标签 −1 从内部评分中剔除。前三行取种子 0、42、2026 的均值。Ward 和 DBSCAN 各引用一次确定性运行，种子 42 仅为接口参数。DBSCAN 的配置来自 B 的参数扫描，是放宽最小簇规模约束后的展示配置。

## 数据来源与计时边界

- K-means、GMM、谱聚类来自 C 的 [统一调用逐种子结果](../results/comparison/shared_three_algorithms/per_seed.csv) 和 [运行清单](../results/comparison/shared_three_algorithms/manifest.json)。K-means/GMM 的 [A 组正式基线](../results/formal/a_shared_baseline_v1/summary.csv) 保留为独立归档，其 fit 耗时采用 estimator.fit 边界。谱聚类采用 C 的 K=4 RBF 对照，独立的 [谱聚类参数探索](spectral_parameter_experiments.md) 保留其原配置与计时。
- Ward 与 DBSCAN 来自 B 的 [正式运行清单](../results/b/full_20261124_v2/manifest.json)、[层次扫描](../results/b/full_20261124_v2/clustering/hierarchical/parameter_summary.csv) 和 [密度扫描](../results/b/full_20261124_v2/clustering/dbscan/parameter_summary.csv)。汇总脚本对照 B 清单校验两张扫描表的文件哈希。

## 五算法同机计时

在同一台机器上使用同一归档矩阵、单线程和同一共享入口。每种算法先预热一次，再按轮次轮换执行顺序，共计时三次。随机算法沿用种子 0、42、2026，Ward 与 DBSCAN 的种子仅作接口占位。墙钟时间覆盖共享入口的验证、线程限制、拟合和标签生成，排除读盘、预处理、评分与写盘。15 次计时输出与各自参考标签的全量 ARI 均为 1。

| 算法 | 同机墙钟秒数（均值 ± 样本标准差） |
|---|---:|
| K-means | 0.0256 ± 0.0011 |
| GMM | 0.1218 ± 0.0086 |
| 谱聚类 | 1.1647 ± 0.0538 |
| Ward | 0.5535 ± 0.0606 |
| DBSCAN | 0.1435 ± 0.0016 |

本机固定配置下，按平均耗时从短到长依次为 K-means、GMM、DBSCAN、Ward、谱聚类。旧三算法墙钟及 B 的原生耗时保留在基准 CSV 中供溯源。B 的 Ward 原生时间包含建树及切树，DBSCAN 原生时间覆盖 estimator.fit。

[耗时图](../results/comparison/shared_five_algorithms/timing/runtime.png) · [逐次计时](../results/comparison/shared_five_algorithms/timing/per_run.csv) · [环境与口径](../results/comparison/shared_five_algorithms/timing/manifest.json)

## 共享接口

- `algorithms.registry.cluster(X, algorithm, params, random_state=42, threads=1)` 支持 `kmeans`、`gmm`、`spectral`、`hierarchical` 和 `dbscan`。返回按输入顺序排列的一维整数标签，DBSCAN 噪声为 −1。X 是已预处理的二维矩阵，CustomerID 在调用侧独立保留。
- K-means、GMM、Ward 和 DBSCAN 的模块同时提供返回 `ClusteringResult` 的 `fit_predict`。统一入口复用这些模块的标签输出。
- 在项目根目录运行 `python -m experiments.five_algorithm_timing` 可复测固定配置。复测前须保留共同子集分析生成的谱聚类全量标签，供计时结果核对。

## 共同非噪声子集评分与一致性

以 B 组选定的 DBSCAN 标签锁定 4,306 位非噪声客户。全部标签来自 4,338 人的全量拟合，按 CustomerID 筛选后在同一矩阵上重评分。K-means、GMM 复用 A 归档，各种子与 C 基准的审计 ARI 均为 1。Ward、DBSCAN 复用 B 归档，重评分与 B 原共同子集结果一致。谱聚类仅补跑基准的三个种子，三项全量评分均与原归档一致，逐客户标签标记为本次补跑。

| 算法 | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---:|---:|---:|
| K-means | 0.2535 ± 0.0006 | 1.2804 ± 0.0021 | 2160.04 ± 0.45 |
| GMM | 0.1081 ± 0.0044 | 2.3516 ± 0.1291 | 1314.25 ± 27.99 |
| 谱聚类 | 0.2578 ± 0.00004 | 1.3359 ± 0.00001 | 2062.05 ± 0.17 |
| Ward | 0.1905 | 1.5445 | 1790.41 |
| DBSCAN | 0.5311 | 0.4393 | 33.39 |

随机算法报告三个种子的均值 ± 样本标准差，Ward 和 DBSCAN 报告单次结果。DBSCAN 两簇仍分别为 4,302 人和 4 人。

种子 42 的跨算法对照中，K-means 与谱聚类最接近（ARI 0.623），DBSCAN 与其他算法的 ARI 均接近 0。共同子集上的跨种子 ARI 范围分别为 K-means 0.947–0.982、GMM 0.881–1.000、谱聚类 0.9993–1.0000。逐客户差异表使用最大重叠一对一匹配，未匹配簇的对应字段留空。交叉表保留所有簇对及人数，可查看拆分和合并关系。

[评分明细](../results/comparison/shared_five_algorithms/common_customers/scores_per_run.csv) · [一致性热图](../results/comparison/shared_five_algorithms/common_customers/ari_seed42.png) · [逐客户标签](../results/comparison/shared_five_algorithms/common_customers/assignments_all_runs.csv) · [簇交叉表](../results/comparison/shared_five_algorithms/common_customers/pairwise_overlaps_seed42.csv) · [匹配后差异](../results/comparison/shared_five_algorithms/common_customers/different_assignments_seed42.csv) · [跨种子 ARI](../results/comparison/shared_five_algorithms/common_customers/ari_between_seeds.csv)

### 实验五的五算法稳定性汇总

| 算法 | 种子 0 / 42 | 种子 0 / 2026 | 种子 42 / 2026 |
|---|---:|---:|---:|
| K-means | 0.9471 | 0.9824 | 0.9624 |
| GMM | 1.0000 | 0.8808 | 0.8808 |
| 谱聚类 | 0.9993 | 0.9993 | 1.0000 |
| Ward | 1.0000 | 1.0000 | 1.0000 |
| DBSCAN | 1.0000 | 1.0000 | 1.0000 |

前三行在共同子集上直接比较三个种子的标签。Ward 和 DBSCAN 复用同机计时复测证据：各自的三次运行均与同一份全量参考划分达到 ARI 1，因此限制到同一共同子集后划分也相同。两种算法的种子参数用于兼容接口。该核验反映固定数据和配置下的复现性。

[五算法稳定性汇总 CSV](../results/comparison/shared_five_algorithms/common_customers/stability_five_algorithms.csv)

在项目根目录运行 `python -m experiments.common_customer_comparison` 可复现该分析，命令会补跑三个谱聚类基准配置。

## 综合判断

在四簇、全覆盖的比较中，谱聚类的平均 Silhouette 比 K-means 高约 0.0054，K-means 的 Davies–Bouldin 最低、Calinski–Harabasz 最高，在本次五算法统一计时中也最快。K-means 适合作为周期性运营分群基线。谱聚类适合进一步检查相似度结构能否带来业务上更清晰的画像；当前稠密 RBF 相似度矩阵随人数平方增长。

Ward 的三项指标均弱于 K-means 和谱聚类，但四个簇分别含 1,158、1,195、921 和 1,064 人。树状结构与较均衡的簇规模便于探索不同分群粒度。GMM 在当前硬标签评分中较弱，若业务需要识别边界客户，可通过归属概率单独评价。

DBSCAN 虽有较高的 Silhouette 和较低的 Davies–Bouldin，但其 4,306 位已分群客户中有 4,302 人落在同一簇，另一簇仅 4 人，还有 32 位噪声客户。在同一 4,306 人子集上，Ward 的 Silhouette 为 0.1905、CH 为 1790.41，DBSCAN 分别为 0.5311 和 33.39。结合规模与覆盖率，这一配置适合探索异常和微型群体，四群运营细分可沿用 K-means 基线。

在项目根目录执行 `python tools/summarize_five_algorithms.py` 可从同一批归档重新生成汇报图和 CSV。

## 实验四：PCA 降维前后对照

对同一份 4,338 人六维标准化矩阵拟合不白化的两维 PCA，PC1 和 PC2 分别解释 60.62% 和 18.55% 的方差，累计 **79.17%**。六维标签复用归档，两维标签通过同一接口按原基准参数拟合。三种随机算法沿用种子 0、42、2026，Ward 和 DBSCAN 各拟合一次。DBSCAN 的 ε 仍为 1.25，用于检验固定配置对表示改变的敏感性。

汇报表将两组标签都放回原六维空间，在所有运行共同分簇的 4,306 位客户上评分。逐次明细还保留各自拟合空间中的原生评分，原生评分的距离与维度随表示而变。

| 算法 | 六维 → PCA2：Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | 前后 ARI |
|---|---:|---:|---:|---:|
| K-means | 0.2535 → 0.2339 | 1.2804 → 1.3610 | 2160.04 → 2097.42 | 0.4164 |
| GMM | 0.1081 → 0.2320 | 2.3516 → 1.3517 | 1314.25 → 2036.31 | 0.3950 |
| 谱聚类 | 0.2578 → 0.2328 | 1.3359 → 1.3185 | 2062.05 → 2031.30 | 0.3784 |
| Ward | 0.1905 → 0.2151 | 1.5445 → 1.3681 | 1790.41 → 1898.18 | 0.5385 |
| DBSCAN | 0.5311 → — | 0.4393 → — | 33.39 → — | 0.0000 |

随机算法表内为三个种子的均值。在共同空间中，GMM 和 Ward 三项指标均改善，K-means 三项均下降，谱聚类有升有降。前后 ARI 为 0.378–0.539 的四簇方案均发生了明显的成员变化。DBSCAN 降维后只剩一个有效簇，含 4,332 人，另有 6 位噪声客户，因此三项内部指标均留空。

[PCA 对照图](../results/comparison/shared_five_algorithms/pca/pca_comparison.png) · [评分明细](../results/comparison/shared_five_algorithms/pca/scores_per_run.csv) · [逐客户归属](../results/comparison/shared_five_algorithms/pca/assignments.csv) · [PCA 载荷](../results/comparison/shared_five_algorithms/pca/loadings.csv)

## 实验六：谱聚类客户画像

使用谱聚类六维 RBF 基准的种子 42 标签，经 CustomerID 连接全量客户原始尺度特征。下表的消费占比按各簇 Monetary 合计计算。

| 簇 | 人数 | 客户占比 | 消费占比 | 最近购买距今中位天数 | 频次中位数 | 消费额中位数 | 客单价中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,192 | 27.48% | 5.38% | 38.5 | 2 | 323.33 | 168.70 |
| 1 | 828 | 19.09% | 2.78% | 263 | 1 | 247.77 | 205.29 |
| 2 | 1,165 | 26.86% | 72.71% | 17 | 7 | 2599.19 | 352.48 |
| 3 | 1,153 | 26.58% | 19.13% | 65 | 2 | 899.63 | 443.60 |

簇 2 购买最近、频次最高，贡献约 72.7% 的消费额。簇 3 的客单价中位数最高，购买频次中位数为 2。簇 0 购买相对近期、客单价中位数最低。簇 1 距上次购买时间最长，频次中位数为 1。这些画像描述观测期内的购买行为。

[原始尺度画像图](../results/comparison/shared_five_algorithms/spectral_profiles/profile_medians.png) · [簇汇总](../results/comparison/shared_five_algorithms/spectral_profiles/cluster_summary.csv) · [六特征统计](../results/comparison/shared_five_algorithms/spectral_profiles/feature_statistics.csv)

## 五算法性能指标热力图

图中三项内部评分使用相同的 4,306 位非噪声客户，统一墙钟耗时取全量 4,338 人的同机三次均值。单元格显示原始均值，颜色按各指标方向在列内排名。DBSCAN 有两个有效簇，簇大小为 4,302 和 4。

[性能热力图](../results/comparison/shared_five_algorithms/performance_heatmap.png) · [绘图数据](../results/comparison/shared_five_algorithms/performance_heatmap_data.csv)

在项目根目录执行 `python -m experiments.complete_c_deliverables` 可重新生成以上交付物。该命令仅重新拟合 PCA 两维固定基线，谱聚类画像和性能热力图复用已归档数据。
