# 电商客户聚类：B/C 公共数据包

数据版本：`online-retail-six-features-v1`。

这份交付完成 A 负责的原始数据清洗和客户特征构建。算法实验尚未开展。

## 直接使用哪些文件

| 成员 | 文件 | 用途 |
| --- | --- | --- |
| B | `data/cleaned_data.csv` | 消费分布、交易行为等探索分析 |
| B、C | `data/customer_features.csv` | 4,338 位客户的六个原始尺度特征 |
| B | `data/customer_dates.csv` | 首次/末次购买日期，辅助解释客户行为 |
| B、C | `docs/data_contract.md` | 清洗口径、字段定义、分析日期和限制 |
| B | `results/tables/customer_cancellation_diagnostics.csv` | 核查客户取消记录，避免把退款相关大额记录直接解释为高价值客户 |
| B、C | `data/dataset_manifest.json` | 数据版本、文件校验值、环境及处理配置 |

**六特征表未做 log1p、标准化或 PCA，这部分由 C 统一处理。CustomerID 只用于对齐结果，不参与聚类。** 诊断表中的取消金额等字段不自动加入约定的六特征。

Python 读取示例：

```python
from pathlib import Path
import pandas as pd

root = Path("Customer_Clustering")  # 根据自己的项目位置修改
features = pd.read_csv(root / "data/customer_features.csv", dtype={"CustomerID": "string"})
feature_columns = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "ProductDiversity", "ActiveDays",
]
customer_ids = features["CustomerID"].copy()
X_raw = features[feature_columns].copy()

transactions = pd.read_csv(
    root / "data/cleaned_data.csv",
    dtype={"CustomerID": "string", "InvoiceNo": "string", "StockCode": "string"},
    parse_dates=["InvoiceDate"],
)
```

客户表按文本 CustomerID 升序排列。算法输出要带回对应 CustomerID，跨表关联按 ID 合并，不凭行号猜测。CSV 使用 UTF-8 BOM，日期使用 ISO 格式，金额单位为英镑。

## 本版统计

| 项目 | 数量 |
| --- | ---: |
| 原始交易明细 | 541,909 |
| 清洗后交易明细 | 392,692 |
| 排除记录 | 149,217 |
| 保留客户 | 4,338 |
| 不同有效正向订单 | 18,532 |
| 分析日期 | 2011-12-10 |
| 正向记录金额合计（GBP） | 8,887,208.894 |

金额保留原数据精度，没有提前四舍五入到分。实际数据存在 0.001 英镑单价。

## 复现

建议 Python 3.12，锁定依赖见 `requirements.txt`。下载包不含原始 Excel，已有交付 CSV 无需重新下载即可使用；复现时从 UCI 官方重新获取原文件。

在本目录运行：

```text
python -m pip install -r requirements.txt
python tools/download_data.py
python prepare_data.py
python -m unittest discover -s tests -v
python tools/verify_delivery.py
```

`prepare_data.py` 也可通过绝对路径从其他工作目录运行。要使用自己的原始文件，可追加 `--input "文件路径.xlsx"`。当前首版全部国家均保留，配置见 `config/data_config.json`。修改配置重跑会更新生成的 CSV 和清单，分享前要一起更新整个数据包。

检查包括 9 项边界测试，以及独立验证程序对保存后的 CSV 做 15 项校验。独立程序不调用清洗或特征计算函数，用 CSV、Decimal、集合和日期重新计算所有客户的六个特征，检查全部原始行的保留/排除去向和文件哈希。结果见 `results/tables/independent_validation.json`。

## 需要共同遵守的口径

- 当前是删除取消行后的正向销售记录口径，**不是扣除退款后的净消费金额**。
- 完全重复行按八个原始字段判断，保留第一行。这是首版分析假设；没有唯一明细 ID，不能证明每个重复行都是错误。配置改为 `keep` 可重建另一口径。
- 不按高消费或固定分位数截断客户。极端订单和特殊 StockCode 已单独列出，细节见数据约定。
- C 统一确定变换、标准化与 PCA 使用方式后，再开始五种算法的正式对比。
- 本包不是已向 B、C 发送的消息，需由组员转交。接口、消融组合等协作约定仍待小组确认。

## 来源

Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
官方页面：<https://archive.ics.uci.edu/dataset/352/online+retail>。
原文件及下载归档的 SHA-256 见 `data/source_manifest.json`；本次分析实际读取文件的 SHA-256 见 `data/dataset_manifest.json`。

## 谱聚类

安装 `requirements.txt` 中的依赖后，在项目根目录运行。默认读取已有的
`data/customer_features.csv`，并在指定输出路径保存每位客户的簇标签。

```bash
python run_spectral.py --n-clusters 4 --affinity nearest_neighbors --n-neighbors 20 --output results/tables/spectral_labels.csv
python run_spectral.py --n-clusters 4 --affinity rbf --gamma 0.1
```

可通过 `--input` 选择其他同格式特征表，通过 `--random-state` 指定种子（默认 42）。
运行入口对 Frequency、Monetary、AvgOrderValue 和 ProductDiversity 执行
`log1p`，再对六列做 Z-score 标准化。CustomerID 用于关联结果。
终端输出簇大小、算法耗时及标准化特征空间中的 Silhouette、Davies–Bouldin 和
Calinski–Harabasz 指标。耗时覆盖建图、谱分解和 K-means。

统一实验框架可调用 `from algorithms.spectral import spectral_clustering`，
传入已经预处理的 `X`（行对应客户，列对应特征），获取与输入行顺序一致的整数标签。
算法构造对称 kNN 图或 RBF 图，计算对称归一化拉普拉斯矩阵，取最小的 K 个
特征值对应的特征向量，将嵌入按行归一化后执行 K-means。kNN 图使用稀疏矩阵，
RBF 图的存储开销随客户数平方增长。图的连通分量数超过 K 时，需增大邻居数
或降低 RBF 的 gamma。簇编号可在不同运行之间置换。
# machine_learning



五种聚类算法可共用 `preprocessing.feature_scaling.load_customer_features(path)`。
统一实验入口加载一次现有的 `data/customer_features.csv`，获得按原表顺序排列的
`customer_ids, X`，再将同一个 `X` 传入各算法。原始尺度特征表也可用于客户画像。
