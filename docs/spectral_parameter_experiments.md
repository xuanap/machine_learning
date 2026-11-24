# 谱聚类参数实验设计与复现

## 准备环境

在 Customer_Clustering 项目根目录执行以下命令。可使用已经安装项目依赖的 Python 环境。

```bash
python -m pip install -r requirements.txt
```

脚本通过 `experiments.inputs.load_input("formal")` 校验
`data/customer_features.csv` 的 SHA-256，再调用公共预处理。使用全部 4,338 位客户，
对 Frequency、Monetary、AvgOrderValue、ProductDiversity 做 log1p，再标准化六列。
CustomerID 用于校验客户顺序。每次调用复用同一份六维矩阵，随机种子为 42，线程数为 1。

三项内部指标都在标准化的六维空间对全部客户计算。Silhouette 和 Calinski–Harabasz
越高越好，Davies–Bouldin 越低越好。耗时以秒计，覆盖建图、特征分解和标签生成。
先检查 `status`、`note`、实际簇数和最小簇占比，再解读指标。小簇需要结合原始消费特征核查。
图的连通分量超过 K 或谱分解未收敛时，保留失败原因，该配置的得分留空。

## 1. 聚类数量 K

分别固定 kNN 的 20 个邻居和 RBF 的 gamma=0.1，在两种图上各运行 K=2–10。
同一张图的构造参数固定，比较同一构造方式下 K 变化带来的差异。

```bash
python -m experiments.parameter_analysis --step k
```

打开 `k.csv`，按 `affinity` 分组，比较各组内九行的三项指标和簇人数。
上次 K 实验给出 K=2，因此后面三项预设 K=2，本次重新计算验证这一依据。

## 2. 相似度构造方式

固定 K=2，比较对称化的稀疏 kNN 图（20 个邻居）与全连接 RBF 加权图（gamma=0.1）。
这两个参数沿用 K 实验的对照配置，结论适用于这两个具体设置。

```bash
python -m experiments.parameter_analysis --step affinity
```

打开 `affinity.csv`，逐项比较两行的分数、簇人数和算法耗时。两行得分应与
`k.csv` 中对应的 K=2 行一致，运行时间允许波动。

## 3. 邻居数量 n_neighbors

固定 K=2 和 kNN 构造，扫描 5、10、15、20、30、50 个邻居。
这组取值覆盖较稀疏到较密的局部连接，保留 20 作为对照点。

```bash
python -m experiments.parameter_analysis --step n_neighbors
```

打开 `n_neighbors.csv`，按邻居数顺序看指标和簇人数。先检查小邻居数下的连通性提示，
再比较有效配置。20 个邻居的得分应与相似度实验的 kNN 行一致。

## 4. RBF 核参数 gamma

固定 K=2 和 RBF 构造，扫描 0.01、0.025、0.05、0.1、0.2、0.5。
取值围绕基线 0.1 分布。gamma 越大，两个客户之间的权重随距离衰减越快。

```bash
python -m experiments.parameter_analysis --step gamma
```

打开 `gamma.csv`，按 gamma 顺序比较指标，并特别检查较大 gamma 是否形成极小簇。
gamma=0.1 的得分应与相似度实验的 RBF 行一致。

## 一次重跑与结果核对

```bash
python -m experiments.parameter_analysis --step all
```

默认输出到 `results/tables/spectral_parameters/`。可追加
`--output-dir results/reproduction/spectral_parameters` 将复跑结果另存。
四张表分别应有 18、2、6、6 行。每个成功配置的簇人数之和应为 4,338。
同名 JSON 记录实际数据和矩阵哈希、代码哈希、参数及软件版本。
复查指标时先对齐数据和代码版本，再比对分数和人数。耗时受运行环境影响，
适合按同机重复计时后再做严格速度比较。

## 本次实验结果

1. 两种相似度构造在各自的 K 扫描中均以 K=2 取得三项内部指标最优。
2. K=2 的基线比较中，RBF（gamma=0.1）的 Silhouette、DB、CH 分别为
   0.3553、1.0830、3081.05，kNN（20 个邻居）分别为 0.3280、1.1590、2603.82。
3. kNN 在本次扫描范围内以 50 个邻居取得三项指标最优，Silhouette 为 0.3396，
   两簇分别有 1,959 和 2,379 人。50 位于本次扫描的上边界。
4. RBF 的 gamma=0.5 取得最高 Silhouette（0.4946）和最低 DB（0.8347），
   但两簇规模为 10 和 4,328 人，CH 由 gamma=0.1 的 3081.05 降至 52.48。
   若目标是形成规模相近的客户分群，可优先保留 gamma=0.1，
   对应两簇为 2,044 和 2,294 人。该选择依据本轮指标与簇规模，尚需客户画像验证。
