# A 的全客户实验结果与解释

本轮于 2026-11-20 运行，使用 4,338 位客户、现有原始六特征和 C 的共享预处理函数。它是实际全客户实验，不是此前 300 人原始尺度 smoke。范围限于 A 的 K-means/GMM；五算法最终表格、DBSCAN 噪声点规则与共同计时仍由 C 整合，RFM/六特征对照是 A 先完成的方案，最终分工仍与 B 确认。

## 数据、预处理与复现

原始特征版本为 online-retail-six-features-v1；CSV SHA-256 为 d031d6b722ae83bf3159a898ede727daa8004a6043acc342dfa87f2609e34013。调用 preprocessing.feature_scaling.load_customer_features，四列 Frequency、Monetary、AvgOrderValue、ProductDiversity 取 log1p，再对六列 StandardScaler。CustomerID 不参与变换，客户顺序保留，基线不使用 PCA 坐标。

种子为 0、42、2026，数值库线程限制为 1。Silhouette 用全量客户与欧氏距离，DB、CH 也使用实际拟合特征空间，定义与 C 当前谱聚类入口一致。耗时只包括 estimator.fit（包括内部初始化），不包括预处理、预测、指标和读写；此耗时暂不直接与未采用相同线程设置的谱聚类横比。

环境为共享 requirements.txt 中的 scikit-learn 1.6.1、SciPy 1.17.0、NumPy 2.3.5、pandas 3.0.1。实验清单另保存 Python 版本、实际参数、代码哈希、矩阵哈希和 model_input.csv。快照以足够精度保存并核对 float64 往返一致，后续距离分析直接读取它，无需 C 另提供标准化 CSV。

在仓库根目录执行：

```text
python main.py --mode formal --suite baseline --output results/formal/新的基线目录
python main.py --mode formal --suite parameters --output results/formal/新的参数目录
python main.py --mode formal --suite ablation --output results/formal/新的对照目录
```

本轮固定结果目录为 results/formal/a_shared_baseline_v1、a_shared_parameters_v1、a_shared_ablation_v1。对应运行数为 6、60、12，共 78 次，全部 status=ok；没有省略失败或未收敛的运行。全部结果中最小簇有 6 人，出现在 GMM K=7、seed=0，说明收敛本身不保证分群适用。

## 相同 K=4、六特征的基线

以下为三个种子的均值；括号内为样本标准差。

| 方法 | Silhouette，越高越好 | DB，越低越好 | CH，越高越好 |
| --- | ---: | ---: | ---: |
| K-means | 0.25044（0.00070） | 1.28770 | 2055.93 |
| GMM，full | 0.10603（0.00460） | 2.38799 | 1255.27 |

在这一输入表示和 K 下，K-means 的三个几何指标较好。GMM 最大化混合分布似然，目标与欧氏距离下的紧凑分群不同，不能把这张表解释为 GMM 在所有任务中更差。三个种子的两两分群 ARI 均值分别为 K-means 0.9638、GMM 0.9206；它们描述这三个已测种子下的稳定性，不是外部真实分类效果。

## 参数变化

K 从 2 到 8：K-means 的 Silhouette 从约 0.3553 降至 0.2225，CH 从约 3084 降至 1600；DB 在本范围内也以 K=2 最小。这支持将 K=2 作为紧凑分群的候选。K=4 则保留更多行为层次，是事先设置的共同基线，不能假称它由指标选出。若需要最终选 K，应同时说明粗分与业务解释粒度的取舍。

GMM full 的平均 Silhouette 在 K=2 时约 0.2794，K=8 时约 0.0739；K=6～8 的种子间波动扩大。相同六维表示下，BIC 在本次 K 网格内偏向 K=8。这表明似然与几何分离目标不一致，也应结合小簇、参数量和实际画像分析，不把 BIC 更低等同于客户分群一定更好。

K-means 初始化对照固定 K=4、n_init=10 和三个种子：random 的平均 Silhouette 为 0.25009，k-means++ 为 0.25044，差异很小。本实验不足以声称某种初始化总能显著改善最终效果。

GMM 固定 K=4 比较协方差类型：full、tied、diag、spherical 的平均 Silhouette 依次为 0.1060、0.1744、0.1314、0.2075；在同一数据表示下，full 的平均 BIC 最低（约 -37183），spherical 的 BIC 约 60989。两种指标对配置的偏好明显不同。六特征存在 Monetary、Frequency、AvgOrderValue 等确定性关系，且单次购买者常有 ActiveDays=0；高置信概率和高似然不等于发现了经过外部验证的真实客户类别。

图表见 results/study/a_shared_v1/kmeans_k_three_metrics.png、gmm_k_three_metrics.png、gmm_covariance_objectives.png。参数运行目录中的 figures 另保存耗时和轮廓系数曲线，并标注有效次数与标准差适用条件。

## RFM 与六特征对照

| 方法 | 特征 | Silhouette 均值 | DB 均值 | CH 均值 |
| --- | --- | ---: | ---: | ---: |
| K-means | RFM | 0.37965 | 0.85920 | 4352.92 |
| K-means | 六特征 | 0.25044 | 1.28770 | 2055.93 |
| GMM full | RFM | 0.08516 | 2.06479 | 1416.54 |
| GMM full | 六特征 | 0.10603 | 2.38799 | 1255.27 |

各列从同一 C 处理矩阵选择，客户、K=4 与种子保持一致。但三个与六个维度的几何空间不同，不能据此直接断言新增特征有害或有益。

进一步比较同一算法的分群划分：seed=42 时 RFM/六特征的 ARI 为 K-means 0.4123、GMM 0.5590，一对一最大重合匹配比例分别约 67.45%、69.39%。这说明新增的客单价、商品多样性和消费跨度改变了哪些客户被归在一起；这些比例不是分类准确率或客户行为随时间的变化率。三种种子的完整记录在 results/study/a_shared_v1/feature_partition_agreement.json。

## 基线客户画像与算法差异

以下选取 seed=42 作为统一展示案例，未根据指标挑选最好种子。统计使用原始单位，Monetary 为未冲抵退款的正向购买金额。群名只是对该次模型的描述，编号没有顺序含义。

| K-means 簇 | 人数 | Recency 中位数，天 | 订单数中位数 | 金额中位数，GBP | 消费跨度中位数，天 | 描述 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1444 | 51 | 2 | 798.60 | 103.5 | 有一定复购与消费跨度 |
| 1 | 798 | 268 | 1 | 300.94 | 0 | 较久未购买、通常单次订单 |
| 2 | 1190 | 17 | 7 | 2609.73 | 308 | 较近期且复购较多 |
| 3 | 906 | 54 | 1 | 217.71 | 0 | 相对近期、通常单次小额订单 |

GMM 则把 1493 位客户归入一个订单数中位数为 1、消费跨度中位数为 0 的簇。它与 K-means 对单次购买客户的拆分方式不同。两算法在相同 seed=42 基线上的 ARI=0.3210、NMI=0.3695，簇匹配后有 2440/4338 人重合（56.25%）；这不是准确率。

K-means 边界示例：CustomerID 13760 的最近/次近中心距离差约 0.000030，CustomerID 16198 约 0.000046，说明在这次拟合空间里接近分群边界。GMM 示例 CustomerID 16112 对两个主要分量的概率约为 0.5277、0.4723，差约 0.0553。

但 GMM 的多数客户最大概率非常接近 1，不能虚构“大量客户同时高概率属于多个群”的故事。概率属于模型内部后验，并未用外部标签校准；同样，不能仅凭金额把极端正向订单客户叫作高价值客户，仍应参照公共数据包的取消订单诊断。

详细画像、边界客户和代表客户保存在 results/analysis/formal/a_shared_baseline_v1；两算法交叉表与簇匹配在 results/comparison/formal/a_shared_baseline_v1。

## 已验证事项及剩余边界

完整测试为 42 项，覆盖新共享函数接入、独立 log1p/Z-score 核对、客户顺序、矩阵快照往返、画像使用实际坐标和全量指标定义。78 次运行保留参数、状态、标签及 GMM 概率；技术结果可以交给 C 整合。

待办包括：B/C 确认消融分工及最终五算法协议，C 整合五种算法表格与共同计时，补充最终汇报和本地个人报告的结果章节。当前技术说明不代替个人报告；个人文稿和讲稿不上传共享仓库。自动源码复核工具缺少 TruffleHog，未完成模型复核，不将测试通过说成自动源码复核通过。
