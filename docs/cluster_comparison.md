# 两种算法的分群对比

## 运行命令

```text
.venv\Scripts\python compare_runs.py --left-experiment results/smoke/a_framework_check_v2 --left-run 000_kmeans_seed42 --right-run 001_gmm_seed42
```

默认右侧使用同一实验目录，也可通过 --right-experiment 指定。两侧必须具有相同的数据版本、源表、实际模型输入、预处理说明和有序特征，客户集合必须完全一致；不静默取交集。只比较 status=ok 的运行，不能混合 smoke 与 formal。

## 输出解释

overlap_counts.csv 是左群为行、右群为列的客户交叉表。overlap_row_fractions.csv 按左群归一化，每行之和为 1。cluster_mapping.csv 使用最大重合的一对一匹配列出右群对应的左群，零重合配对不当作有效对应。

customer_comparison.csv 按客户编号关联两种标签，保存匹配后的对应和是否重合。different_assignments.csv 仅列出匹配后不重合的客户。comparison.json 保存 ARI、NMI、匹配重合比例、未匹配群体及数据/代码来源。

所有这些统计描述分群一致性，不是分类准确率。标签编号互换不会改变划分；群体拆分、合并或两侧簇数不同，可能让一对一匹配留下未对应群体。不能把该情况直接说成客户行为真实变化。

## 测试与后续使用

测试覆盖纯标签置换、客户顺序变化、簇数不等、零重合配对、客户集合不一致和重复编号。正式实验产生两种标签后，可直接使用相同入口生成比较；当前示例仍标记 SMOKE ONLY。
