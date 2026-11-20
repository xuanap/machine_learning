# 客户画像与边界客户分析

此阶段完成分析工具，当前全客户基线已完成，结果见 docs/a_experiment_results.md。工具读取已保存结果，不重新拟合模型，也不自行标准化。

## 运行方式

先验证已有 smoke 运行：

```text
.venv\Scripts\python analyze_run.py --experiment results/smoke/a_framework_check_v2 --run-id 000_kmeans_seed42
.venv\Scripts\python analyze_run.py --experiment results/smoke/a_framework_check_v2 --run-id 001_gmm_seed42
```

正式运行后，把 --experiment 换成实际正式结果目录、--run-id 换成实际运行编号。新实验自动读取实验目录中的 model_input.csv；只有没有该快照的旧版结果才需要用 --matrix-input 指定原拟合 CSV。

该 CSV 必须与拟合时的文件哈希一致；距离只能在模型实际使用的坐标里计算，不能拿原始金额/天数去和标准化后的中心相减。

## 输出

| 文件 | 内容 |
| --- | --- |
| customer_assignments.csv | 客户编号、分群、原始六特征、对应的边界诊断 |
| cluster_summary.csv | 各群人数、客户占比、正向金额合计与占比、六特征均值和中位数 |
| feature_statistics.csv | 按群、按特征的长表统计；单客户群的样本标准差保留缺失 |
| boundary_customers.csv | 两个群归属最接近的最多 10 位客户，仅用于人工分析 |
| representative_customers.csv | 每群最多 3 位模型内较典型客户，不等于业务价值最高客户 |
| analysis_manifest.json | 数据、拟合运行和分析代码来源及哈希 |
| NOTICE.txt | smoke 或正式运行的适用边界 |

画像统计始终使用原始尺度六特征。各群占比的分母是本次运行涉及的客户，不自动代表整个数据集；smoke 抽取 300 人时不能冒称 4,338 人的总体画像。

## 两种算法各分析什么

K-means：计算到已分配中心、最近中心和次近中心的距离。DistanceGap 为次近减最近距离，小差距表示在本次模型下更靠近分群边界。不同特征组合、不同尺度下的距离差不能直接横比。

GMM：读取每客户归属概率，按客户编号关联，验证概率归一化和最大概率对应已有标签。输出最高概率、第二概率、两者差以及归一化熵。ProbabilityGap 越小、熵越高，说明本模型的归属更不明确，但不代表概率已经过外部校准，也不证明真实业务群体有重叠。

单成分模型没有第二个群，差距字段保留缺失，不制造“确定性”结论。各群命名应结合统计和取消记录人工判断，工具不自动命名“高价值”“流失”等群体。

## 需要保持的业务口径

Monetary 来自清洗后的正向销售记录，未冲抵退款。尤其是极端金额客户，应结合原数据交付中的取消诊断核查，不能仅因金额大就称为高价值客户。

只有 status=ok 的运行可进入该分析入口。未收敛和失败结果应先处理，不能用画像包装成正常最终结果。smoke 分析的输出和通知明确保留 smoke 标识。
