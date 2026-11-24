# 统一聚类对比实验

## 统一调用

`algorithms.registry.cluster(X, algorithm, params, random_state=42, threads=1)`
接收预处理后的二维特征矩阵、算法名和算法参数，返回每个输入客户对应的整数标签。
K-means 和 GMM 复用 A 的实现，谱聚类调用 C 的实现。
接入 DBSCAN 和层次聚类时，在 `ADAPTERS` 增加适配器，接收
`(X, params, seed, threads)` 并返回标签，保持客户行顺序。DBSCAN 的 -1 表示噪声。

## 数据、评分和计时

正式运行校验原始特征表，调用一次公共预处理核对坐标，使用全部 4,338 位客户的六维标准化特征。
三种算法共用 K=4、种子 0/42/2026 和单线程，沿用 A 的 K-means/GMM 基线参数，
谱聚类使用 RBF gamma=0.1。K=4 用于与 A 的同粒度基线对齐。

统一评估函数 `experiments.group_metrics.evaluate_comparison` 计算全量已分簇客户的
欧氏 Silhouette、Davies–Bouldin 和 Calinski–Harabasz，同时报告有效簇数、噪声人数和覆盖率。
含噪声算法的分数应结合覆盖率阅读，只有一个有效簇或每人一个簇时指标留空。
通用对比耗时测量统一调用的墙钟时间，包含验证、拟合和标签生成，
排除预处理、指标计算和文件写入。汇总取三个种子的均值和样本标准差。

## 接入 A 的结果与复现

脚本读取 A 的正式基线归档，核对输入矩阵指纹和各次运行参数。为避免不同环境的浮点舍入影响，
三种算法均使用通过哈希校验的 A 归档标准化矩阵。归档输入与公共预处理按客户顺序逐项核对，
最大绝对差写入 manifest。通过 CustomerID
对齐原标签并用统一函数重新评分。归档标签和重跑标签的 ARI 用于判断两次分组是否一致，
无需对齐簇编号。综合表使用同机重跑结果，以统一计时边界。

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
python -m experiments.baseline_comparison
```

通过 `--archive` 指定其他同结构 A 基线归档，通过 `--output` 指定输出目录。
默认输出在 `results/comparison/shared_three_algorithms/`。

- `summary.csv` 汇总三种算法的指标与耗时。
- `per_seed.csv` 保存逐种子结果。
- `a_archive_audit.csv` 保存 A 的原分数、重评分和重跑 ARI。
- `heatmap.png` 标注各列原始均值，底色按各指标优劣方向在列内排名，颜色越深排名越高。
- `manifest.json` 保存输入溯源、参数、种子及口径。
