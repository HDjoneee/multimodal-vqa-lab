# 项目待办清单

按「阻塞提交 / 低成本高价值 / 可选改进 / 写作准备 / 仓库卫生」五档排列。每项都写了验收标准和预估成本，做完勾掉。

状态标记：`[ ]` 未开始，`[~]` 进行中，`[x]` 完成。

---

## A. 阻塞提交的硬性要求

### A1. 计算量与延迟统计

要求原文：「统计 Params、MACs 或 FLOPs（注明口径）及推理耗时或显存」。这是并联的两组，都要有。

- [ ] 新建 `scripts/benchmark_cost.py`
  - 移植 `scripts/experiment_source.py:425` 的 `clip_macs()` 解析式计数
  - 参数：CLIP 图像编码器 + 文本编码器 + 分类头的总参数量、各组可训练参数量
  - MACs 口径要在报告里写明：只算卷积、线性层、注意力矩阵乘，不含 LayerNorm、激活、预处理、磁盘 IO；图像 224×224、文本 77 token
  - 延迟：单样本前向中位数，预热 5 次、测量 30 次，分别报 CPU / MPS / CUDA
  - 显存：`torch.cuda.max_memory_allocated()` 峰值
- [ ] 在 RTX 4060 上跑三组（text / fusion / finetune），输出 `results/cost_table.json` 与 markdown 表
- [ ] 报告里给一张表：Params / MACs / 单样本延迟 / 峰值显存，三组对照

**验收**：三组 × 四项指标齐全，MACs 口径成文，数字可复现。
**成本**：半天。

### A2. 英文 IEEE 格式课程论文

- [ ] 确认模板版本（IEEE conference 或 transactions）与页数限制
- [ ] 搭骨架，六个一级标题：Introduction / Related Work / Method / Experiments / Results and Discussion / Conclusion and Limitations
- [ ] 图表清单定稿（见 D2），先占位再填写
- [ ] 正文写作，顺序：Method → Results → Experiments → Related Work → Introduction → Conclusion
- [ ] 参考文献套用 `docs/related_work.md` 里的 BibTeX
- [ ] 语法与格式通检

**验收**：PDF 编译通过，图表编号连续，引用无缺项。
**成本**：主要工作量，留足时间。

### A3. 演示录屏或截图

- [ ] `demo/three_way.html` 录一段 30–60 秒，展示同一道题三组答案与置信度差异
- [ ] 另存 3–5 张关键截图，放进论文 Figure

**验收**：视频能播，截图不超过 5 张且每张都在正文被引用。
**成本**：半小时。

---

## B. 低成本高价值

### B1. 补 F1 指标

- [ ] 在 `three_way_experiment_source.py` 与 `top1000_experiment_source.py` 的评估段加 macro-F1 / weighted-F1
- [ ] 预期：micro-F1 = Accuracy；macro-F1 会明显偏低，正好量化长尾与词表死类问题
- [ ] 报三个口径：micro / macro / weighted

**验收**：三组 × 三个 F1 进 `results/*_summary.json`，报告 Limitations 引用 macro-F1。
**成本**：一小时。

### B2. 长尾结构的量化补充

- [ ] 统计训练集每类样本数分布（已有：467 类 <5，中位数 4，`yes`+`no` 占 42%）
- [ ] 统计评估集按类召回率，画一条「按类样本数分桶的准确率」曲线
- [ ] 与 macro-F1 互相印证

**验收**：一张分桶柱状图，能支撑「准确率由 117 个头部类决定」这个结论。
**成本**：两小时。

---

## C. 实验改进（按文献支撑强度排序，可选）

### C1. 软交叉熵损失

- [ ] 依据：Ilievski & Feng, arXiv:1708.00584。训练用众数硬标签、评测用 10 票软分，两者错配
- [ ] 改 `F.cross_entropy(head(...), y)` 为对 10 个答案的票数加权交叉熵
- [x] 数据已就绪：manifest 的 `human_answers` 字段存了全部 10 条，20,989 行无一缺失，无需重新生成数据
- [ ] 在 `v3-20989-3-aligned` 的三组上重跑
- [ ] 预期：+1.2~1.6%（原文 AVG +1.2、POOL +1.6）

**验收**：三组相对基线有提升，且验证 loss 不再「loss 涨、准确率也涨」。
**成本**：改 15 行 + 20 分钟训练。

### C2. LP-FT 两阶段

- [ ] 依据：Kumar et al., ICLR 2022 Oral
- [ ] 拿已训好的 fusion 头，解冻末层再续训 3 轮
- [ ] 复用主干缓存，一分钟内出结果
- [ ] 若有效，把「微调 −0.8」改写成「微调方式错了，不是微调无效」

**验收**：第四组结果进表，与 finetune 组对照。
**成本**：半小时。

### C3. 换 CLIP-ResNet 主干

- [ ] 依据：Shen et al., ICLR 2022。ViT 检测定位 AP 0.03，ResNet 3.14
- [ ] `clip.load("RN50")`，重提特征
- [ ] 重点看数量题与颜色题

**验收**：数量题准确率是否超过当前 29.9%。
**成本**：下载权重 + 重提特征约一小时。

### C4. 特征尺度与 LayerNorm

- [ ] `FEATURE_SCALE = math.sqrt(512)` 是我们自己加的，原为 CLIP cosine 分类头设计
- [ ] 分类头前加 LayerNorm，或把 scale 降到 1
- [ ] 作为 sanity check，预期 +1~2

**验收**：三组同步涨则说明问题在文本侧尺度；盯着 fusion−text 差值是否扩大。
**成本**：改三行 + 十分钟。

### C5. Patch 特征加权池化

- [ ] 现在只用 CLIP 的单一 CLS 向量，49 个 patch token 全部丢掉
- [ ] 取出 7×7 个 patch 向量，让分类头学加权求和
- [ ] 数量题、空间关系题受益最大

**验收**：数量题有提升；需改特征缓存格式。
**成本**：约 100 行。

---

## D. 写作准备

### D1. Related Work 素材

- [x] `docs/related_work.md` 已成稿，11 篇含 BibTeX

### D2. 图表清单

| 编号 | 内容 | 数据来源 | 状态 |
| --- | --- | --- | --- |
| Fig. 1 | 整体流程（编码 → 融合 → 分类头） | 需新画 | `[ ]` |
| Fig. 2 | 数据集抽样与三个 split 的划分关系 | manifest stats | `[ ]` |
| Fig. 3 | 训练曲线（三组 loss 与选模准确率） | `results/three_way_training_curves.png` | `[x]` |
| Fig. 4 | 按答案类型分组的准确率柱状图 | `results/three_way_by_type.png` | `[x]` |
| Fig. 5 | 演示截图（同题三组答案） | `demo/three_way.html` | `[ ]` |
| Table 1 | 数据规模与分布对照（官方 / 训练 / 开发 / 测试） | `verify_data.py` | `[x]` |
| Table 2 | 三组主结果（EM / soft / 两个分母口径） | `results/three_way_summary.json` | `[x]` |
| Table 3 | 分题型准确率 | 同上 | `[x]` |
| Table 4 | 计算量与延迟 | A1 产出 | `[ ]` |

### D3. 必须核对后再入文的数字

- [ ] 词表内 EM：text 35.9 / fusion 41.6 / finetune 40.8（`v3-20989-3-aligned`）
- [ ] 主交付：fusion 42.6（`v3-20989-2-main`，16 轮）。**与 41.6 不可混用，引用须标来源**
- [ ] 数量题图像增益：`-2-main` 为 0，`-3-aligned` 为 +4.7，报告里选一个并说明口径
- [ ] 开发集 8,083 词表内 / 8,969 全量；测试集 8,072 / 8,919
- [ ] 训练覆盖 900/1000 类，词表死类 100 个
- [ ] 旧版 12,400 与新版 20,989 的对比只能说方向，不可跨开发集比绝对值

### D4. 待确认的提交要求

- [ ] 论文页数与模板具体版本
- [ ] 演示视频时长与格式
- [ ] 是否需要提交模型权重（`results/*_head.pt` 未入库）
- [ ] 组员分工与贡献声明怎么写

---

## E. 仓库卫生

- [ ] `results/archive_12400qa/` 已在 gitignore，确认不需要入库
- [ ] `requirements.txt` 现在全是 `>=` 无上限（`torch>=2.2`），论文附录需写明实际版本：本地 MPS 用 torch 2.12，4060 用 2.14.1+cu126
- [ ] 本地 `.venv` 与远端 `D:\vqa\venv` 两套环境要分别记录，附复现命令
- [ ] 写一个小脚本校验 README 里的数字与 `results/*.json` 一致，避免改实验后忘记同步
- [ ] `docs/` 目录目前只有 `related_work.md`，论文成稿后放这里

---

## 建议执行顺序

1. A1 计算量与延迟（硬性，不做直接扣分）
2. B1 F1 指标（低成本，能进 Limitations）
3. C1 软交叉熵（唯一可能真正提分的改动）
4. A2 论文骨架 + 正文
5. C2 LP-FT（若论文需要「微调为何失败」的正面答案）
6. A3 录屏（最后做，避免演示内容随实验变动）
