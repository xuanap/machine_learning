"""Render a Chinese experiment interpretation from the actual saved results."""
from preprocessing.feature_engineering import FEATURE_COLUMNS


def profile_lines(table):
    lines = []
    for _, row in table.iterrows():
        name = "噪声组" if row.Cluster == -1 else f"簇 {int(row.Cluster)}"
        values = "，".join(f"{column}={row[f'{column}_median']:.2f}" for column in FEATURE_COLUMNS)
        lines.append(f"  {name}：{int(row.Customers)} 人，占 {row.CustomerShare:.2%}，正向销售额占 {row.PositiveSalesShare:.2%}；六特征中位数：{values}。")
    return lines


def write_report(path, eda=None, clustering=None):
    lines = ["B 成员实验结论（全量数据）", "",
             "口径：金额单位为 GBP；保留正向交易，未与取消/退款订单冲抵，不能称为净消费。",
             "CustomerID 仅用于关联，不进入模型。画像均使用原始尺度六特征，标签编号没有固定业务含义。", ""]
    if eda is not None:
        summary, stats, pairs, top = eda
        lines += ["第一步：消费行为探索",
                  f"共 {summary['detail_rows']:,} 条交易明细，按 CustomerID + InvoiceNo 汇总为 {summary['orders']:,} 笔订单，覆盖 {summary['customers']:,} 位客户。",
                  f"正向销售额 {summary['positive_sales_gbp']:,.3f} GBP；订单金额中位数 {summary['order_amount_median']:,.2f}，P99 为 {summary['order_amount_p99']:,.2f}，最大值 {summary['order_amount_max']:,.2f} GBP。订单金额明显右偏，少数极端订单把均值抬高。",
                  f"观测期内仅买过一次的客户 {summary['single_order_customers']:,} 人，占 {summary['single_order_fraction']:.2%}。这是窗口内单次购买，不能据此认定终身只买一次或已经流失。",
                  f"消费集中度 Gini={summary['gini']:.4f}。"]
        for row in top.itertuples():
            lines.append(f"前 {row.TopCustomerFractionRequested:.0%} 客户（向上取整为 {row.Customers} 人）贡献 {row.SalesShare:.2%} 正向销售额。")
        lines += [f"月度订单数和销售额见 eda/monthly.csv；不完整月份：{', '.join(summary['incomplete_months'])}。",
                  "观测范围按数据清单的首末日判断，首日是月初时不因首笔交易晚于午夜而标成缺月。2011-12 只覆盖 1—9 日，其总量不能直接与完整月份比较；不据此推断当月需求下降。", "",
                  "第二步：六特征分布与相关性"]
        for _, row in stats.iterrows():
            lines.append(f"{row.Feature}：中位数 {row['50%']:.2f}，均值 {row['mean']:.2f}，P99 {row['99%']:.2f}，最大值 {row['max']:.2f}，偏度 {row['skew']:.2f}。")
        lines += ["Frequency、Monetary、AvgOrderValue、ProductDiversity 存在长尾，原始尺度直方图被少数大值拉伸；同时给出 log1p 视图便于看主体。图中的六列 log1p 仅作可视化，模型严格采用 C 的四列 log1p 与六列 Z-score，不对 Recency/ActiveDays 追加变换。",
                  "Pearson 衡量原始尺度线性关系，易受极端客户影响；Spearman 衡量秩的单调关系。以下列出任一种相关系数绝对值达到 0.70 的特征对："]
        pair_explanations = {
            frozenset(["Frequency", "Monetary"]): "累计金额包含订单数这一乘数；客单价差异及极端大单会削弱原始线性关系。",
            frozenset(["Frequency", "ProductDiversity"]): "更多订单提供积累不同商品编码的机会，但重复购买同一编码不会增加多样性。",
            frozenset(["Frequency", "ActiveDays"]): "首末购买跨度长的客户通常有更多复购机会；跨度不是实际下单天数。",
            frozenset(["Monetary", "ProductDiversity"]): "购买范围广通常伴随较多累计交易；少编码的大宗订单可能偏离这一趋势。",
            frozenset(["Monetary", "ActiveDays"]): "更长的购买跨度提供累积金额的机会，金额也受订单次数和客单价影响。",
            frozenset(["ProductDiversity", "ActiveDays"]): "更长的购买跨度提供接触和积累不同编码的机会。",
            frozenset(["Monetary", "AvgOrderValue"]): "金额定义含客单价这一乘数；单次极大订单及其取消记录尤其需要核查。",
            frozenset(["Recency", "ActiveDays"]): "两者共用末次购买日期，末次日期越晚会减小 Recency，并可能拉长首末跨度。",
        }
        for row in pairs[pairs.MaxAbsoluteCorrelation >= .7].itertuples():
            explanation = pair_explanations.get(frozenset([row.Feature1, row.Feature2]), "这反映观测期内购买行为的共同变化，仍需结合定义和异常值解释。")
            lines.append(f"  {row.Feature1} / {row.Feature2}：Pearson={row.pearson:.3f}，Spearman={row.spearman:.3f}。{explanation}")
        lines += ["Monetary = Frequency × AvgOrderValue 是定义上的关系；购买次数、累计金额、编码多样性和活跃跨度还共同反映购买机会与参与度。相关性不证明因果，也不代表六列完全独立。Recency 以 2011-12-10 为分析日，越低通常表示购买越近期；ActiveDays 是首末购买跨度而非实际购买天数。ProductDiversity 还包括保留的运费等特殊编码，不严格等于实体商品种类数。",
                  "极端客户核查见 eda/extreme_customer_cancellation_audit.csv，并复制了仓库已有的取消候选表供追溯。客户 16446 的 168,469.60 GBP 正向明细随后有 C581484 候选取消；客户 12346 的 77,183.60 GBP 正向记录也有对应候选取消。不能直接把这些客户称为高价值忠诚客户。取消绝对金额比值只作诊断，候选匹配不是已验证的订单级冲抵，未据此更改六特征或计算净额。", ""]
    if clustering is not None:
        h, d = clustering["hierarchy"], clustering["selected"]
        baseline = h[(h.linkage == "ward") & (h.k == 4)].iloc[0]
        lines += ["第三步：层次聚类",
                  "从 C 的 load_customer_features 加载一次 4,338×6 矩阵，两种算法使用同一对象，无抽样、PCA 或再次标准化；精确坐标见 clustering/model_input.npy 和带 CustomerID 的 model_input.csv。",
                  f"先跑 Ward K=4 基线，Silhouette={baseline.silhouette:.4f}，Davies–Bouldin={baseline.davies_bouldin:.4f}，Calinski–Harabasz={baseline.calinski_harabasz:.2f}。",
                  "再比较 Ward/Complete/Average 的 K=2..8，共 21 个配置；三种树状图均由全量客户构造，只截断显示最后 35 个合并组，不是抽样拟合。簇规模和评价详见 clustering/hierarchical/parameter_summary.csv。"]
        for method, group in h.groupby("linkage", sort=False):
            best = group.sort_values("silhouette", ascending=False).iloc[0]
            lines.append(f"{method} 在扫描中最高轮廓系数对应 K={int(best.k)}，Silhouette={best.silhouette:.4f}，最小/最大簇为 {int(best.min_cluster_size)}/{int(best.max_cluster_size)} 人。")
        lines += ["轮廓系数较高仍须看簇规模：Complete/Average 可能以很小的极端簇换来较高分数，不能只按一列指标宣布业务最优。Ward 倾向降低簇内平方和，四簇画像为："]
        lines += profile_lines(clustering["ward_profile"])
        wp = clustering["ward_profile"]
        recent = wp.sort_values(["Recency_median", "Frequency_median"], ascending=[True, False]).iloc[0]
        frequent = wp.sort_values("Frequency_median", ascending=False).iloc[0]
        dormant = wp.sort_values("Recency_median", ascending=False).iloc[0]
        lines += [f"结合中位数看：簇 {int(recent.Cluster)} 购买较近期；簇 {int(frequent.Cluster)} 的典型购买次数最高；簇 {int(dormant.Cluster)} 距离最近一次购买最久。可以据此讨论活跃度、复购和唤回候选群，具体金额判断须同时审阅取消统计。K=4 是规定基线，不能从这一实验推断真实客户天然恰有四类。", "",
                  "第四步：DBSCAN",
                  "先绘制 min_samples=5/10/20 的近邻距离曲线。邻域计数含样本自身，故使用包含自身的第 min_samples 个近邻距离。eps 使用预先固定的 0.25/0.35/0.45/0.55/0.65/0.8/1/1.25/1.5/2，与三个 min_samples 组合共扫描 30 组。",
                  "所有标签保留 -1；噪声不作为一个有效簇。内部指标仅对非噪声客户精确计算；有效簇不足两个，或簇数等于样本数时，指标保存为空值并记录原因，不填成 0。",
                  f"展示参数 eps={d.eps:g}, min_samples={int(d.min_samples)}：有效簇 {int(d.effective_clusters)} 个，覆盖率 {d.coverage:.2%}，噪声 {int(d.noise_customers)} 人（{d.noise_fraction:.2%}），拟合耗时 {d.fit_seconds:.4f} 秒。",
                  f"选择阶段：{clustering['selection']['stage']}。预定规则先要求覆盖率≥80%、至少两簇、最小簇≥max(20,ceil(0.5%×客户数))，再按轮廓系数排序；无解时依次放宽最小簇规模和覆盖率，详见 selection.json。"]
        if clustering["selection"]["stage"] != "coverage_and_cluster_size":
            lines.append("本次未满足全部首选条件，因此所选参数属于探索性展示；不能将其描述为稳健的最优客户分群。")
        lines += profile_lines(clustering["dbscan_profile"])
        lines += ["噪声组原始尺度的六特征统计、人数、销售额占比和取消统计另存于 clustering/dbscan_selected/noise_*.csv。噪声仅表示当前密度参数下未被吸收，可能包括大额、低频或稀有组合客户，不能等同无价值客户或错误数据。", "",
                  "同一批客户的评价与限制",
                  "Silhouette/CH 越大通常越好，DB 越小通常越好；这些均为内部几何指标，没有外部真实类别验证。两种算法在各自评价人群上的原始分数只分开报告（metrics_with_separate_scopes.csv），不直接横比。以下在所选 DBSCAN 的非噪声客户上重新计算两种算法："]
        for row in clustering["common"].itertuples():
            score = f"Silhouette={row.silhouette:.4f}, DB={row.davies_bouldin:.4f}, CH={row.calinski_harabasz:.2f}" if row.metrics_available else "指标不可用：" + row.metric_note
            lines.append(f"  {row.algorithm}：共同客户 {row.common_customers} 人，共同样本占全体 {row.common_fraction_of_all_customers:.2%}；算法原始覆盖率 {row.original_coverage:.2%}；{score}。")
        sizes = clustering["ward_profile"].Customers
        lines += [f"面向客户分层，Ward 四簇规模为 {int(sizes.min())}—{int(sizes.max())} 人，并呈现近期复购、普通复购及较久未购的差异，较便于讨论运营动作；但其全量轮廓系数仅 {baseline.silhouette:.4f}，不能称为边界清晰的天然群体。"]
        if d.evaluated_customers and d.max_cluster_size / d.evaluated_customers > .95:
            lines.append(f"所选 DBSCAN 的最大簇包含非噪声客户的 {d.max_cluster_size / d.evaluated_customers:.2%}，最小簇仅 {int(d.min_cluster_size)} 人；较高轮廓系数主要伴随主体客户与极小稀有群的分离，未形成均衡的客户层级，因此不据此判定它优于 Ward。")
        noise = clustering["dbscan_profile"]
        noise = noise[noise.Cluster.eq(-1)]
        if not noise.empty:
            lines.append(f"噪声客户的正向销售额占比为 {noise.PositiveSalesShare.item():.2%}，应作为稀有行为和取消记录的重点核查对象，而不是直接删除。")
        lines += ["这是对已有标签限制到共同子集后的重评，没有在子集重新拟合；共同子集由 DBSCAN 决定，不能推断被排除噪声客户上的效果。拟合计时只覆盖树构建与切割，或 DBSCAN.fit，不包括读写、预处理、指标和画图；层次扫描的每组均独立建树，Ward K=4 复用已先运行的基线结果。",
                  "两种方法对偏态、密度差异和极端客户的反应不同。优先结合覆盖率、最小簇规模、原始画像及业务可行动性解释，不能仅凭最高轮廓系数选算法。", ""]
    lines += ["复现方式（项目根目录，输出路径必须是 results/b 下尚不存在的新目录）：",
              r".\.venv\Scripts\python.exe -m pip install -r requirements.txt",
              r".\.venv\Scripts\python.exe run_b.py --mode eda",
              r".\.venv\Scripts\python.exe run_b.py --mode clustering",
              r".\.venv\Scripts\python.exe run_b.py --mode all",
              r".\.venv\Scripts\python.exe -m unittest discover -s tests -v",
              r".\.venv\Scripts\python.exe tools/verify_b_run.py results/b/本次输出目录",
              "数据、共享预处理和 B 代码哈希、环境版本、矩阵哈希及输出校验值见 manifest.json。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
