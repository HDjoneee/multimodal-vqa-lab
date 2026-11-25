# VQA v2 图文视觉问答课程实验

当前版本在同伴整理的 Top‑1,000 数据集上实际训练了仅文本与冻结 CLIP 图文模型，提供 [已运行的 Top‑1,000 Notebook](notebooks/VQA_Top1000.ipynb) 和 [HTML 问答案例](demo/top1000.html)。历史四题型的 [实验 Notebook](notebooks/VQA实验.ipynb) 和 [网页演示](demo/demo.html)也保留。原始数据、模型权重和图片存放在本地。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 数据下载和预处理、实验源码、Notebook 与演示生成脚本 |
| `notebooks/` | 实验 Notebook |
| `demo/` | HTML 演示和模板 |
| `results/` | 预测、指标和图表 |
| `data/`、`vendor/` | 本地数据、模型权重与第三方实现，按需生成 |

## 数据与模型

- 问题、人工答案： [VQA v2 官方下载页](https://visualqa.org/download)，使用 real images 的 train/val 标注和问题文件。
- 图片： [COCO 2014](https://cocodataset.org/)，只下载清单中引用的图片。COCO [使用条款](https://cocodataset.org/dataset/termsofuse.htm)说明图片版权属于原上传者；本仓库的 HTML 引用官方图片地址，不附带图片副本。
- 预训练模型与分词实现： [OpenAI CLIP](https://github.com/openai/CLIP)，ViT-B/32。权重 SHA256 在代码中验证；运行 `scripts/download_sources.py` 后第三方实现会位于 `vendor/CLIP-main`，并保留原许可。

## Top‑1,000 扩容数据集

按完整官方训练集的 `multiple_choice_answer` 频次建立全局 Top‑1,000 词表，同频答案按字母顺序排序。对官方训练集的全部题型先过滤词表外的 QA，再按图片统计保留的 QA 数。每张候选图至少有 6 条高频 QA。默认从恰有 6 条和 7 条的两档按 4:1 抽样，并保留入选图片的全部词表内 QA。

训练集包含 **2,000 张训练图片、12,400 条训练 QA**。开发集沿用同一词表及答案 ID，从官方 val 取 **800 张图片、4,960 条 QA**，其中 640 张图各有 6 条高频 QA，160 张各有 7 条。开发图片池沿用种子 42 的历史划分。

默认抽样种子为 42。训练清单保存在 `data/manifest_top1000.json`，开发清单保存在 `data/manifest_top1000_dev.json`，两者分别引用 `data/images/train2014/` 和 `data/images/val2014/` 中的图片。先生成训练清单，再生成开发清单。只生成清单可省略 `--download-images`。

```powershell
python -m pip install -r requirements.txt
python scripts/download_sources.py
python scripts/prepare_data.py --sampling top1000 --download-images
python scripts/prepare_data.py --sampling top1000-dev --download-images
python scripts/build_notebook.py --source scripts/top1000_experiment_source.py --output notebooks/VQA_Top1000.ipynb --execute
python scripts/export_top1000_demo.py
```

默认执行 `python scripts/prepare_data.py` 会生成 Top‑1,000 训练清单。Notebook 使用全局 1,000 类分类头，选出开发集表现较好的模型，生成 `results/top1000_summary.json`、`results/top1000_predictions.json` 和两张图。训练后的分类头保存在本地 `results/top1000_head.pt`，不提交 Git。只生成 Notebook 结构可省略 `--execute`，但其中不会出现真实结果。仓库版 HTML 引用 COCO 官方图片地址，打开时需要网络。

默认使用本仓库的 `data/` 和 `vendor/`。如需复用同一台电脑上的已有公开数据与 CLIP 实现，可设置 `VQA_DATA_ROOT` 和 `VQA_VENDOR_ROOT` 指向相应目录；未设置时，以上命令独立运行。

## 调参和官方评测

训练集与开发集按官方 train/val 图片隔离。开发集 800 张图按图片 ID 再分成 600 张模型选择图片和 200 张置信度校准图片；两组的词表内 QA 总共 4,960 条。同一批图片有 5,547 条未按答案过滤的 QA，答案词表覆盖率为 89.4%。题型标注只用于结果分组，不用来限制模型输出。训练词表只来自官方 train 标注。

Top-1,000 训练子集只包含词表内题；未过滤开发集保留全部 587 道词表外题。在 Exact Match 中，这些题只有模型预测到其标注答案才能计为正确，而封闭词表模型无法输出该答案，因此计为错误。VQA-style 评分仍逐题与 10 个人工答案比较。

Notebook 同时报告词表内与未过滤开发题的 Exact Match，并依据 10 个人工答案计算带短答案规范化的 **VQA-style** 分数。这是本地实现的开发集分析指标；正式 [VQA Accuracy](https://visualqa.org/evaluation.html) 应以官方评测结果为准。当前所有新结果均为开发集成绩，不能称为 VQA v2 官方测试成绩。模型定稿后，可按 [官方结果格式](https://visualqa.org/evaluation.html) 导出 `question_id`、`answer` JSON 并提交 Test-Dev。

### 本次 Top-1,000 真实运行结果

设备：NVIDIA GeForce RTX 4060 Laptop GPU；分类头最多训练 16 轮，按 600 张开发图片上的 3,719 道词表内题选最佳轮次与模型。另有 200 张图片、1,241 道词表内题仅用于温度缩放。下表的词表内成绩基于全部 4,960 道开发题，未过滤成绩基于同一批图片的全部 5,547 道题。

| 模型 | 词表内 Exact Match | 未过滤 Exact Match | 未过滤 VQA-style |
| --- | ---: | ---: | ---: |
| 训练集多数答案 `no` | — | 19.1% | 24.8% |
| 仅文本（最佳第 15 轮） | 34.0% | 30.4% | 38.1% |
| 冻结 CLIP 图文融合（最佳第 11 轮） | **39.1%** | **35.0%** | **42.5%** |

未过滤开发题的 Exact Match 按官方 `answer_type` 分组：

| 答案类型 | 题数 | 仅文本 | 图文融合 |
| --- | ---: | ---: | ---: |
| 是非 | 2,062 | 51.8% | 57.1% |
| 数量 | 793 | 23.7% | 23.7% |
| 其他 | 2,692 | 15.9% | 21.3% |

图文模型在未过滤开发题上比仅文本高 4.6 个百分点。数量题没有 Exact Match 增益。此次按每张图保留的高频 QA 数抽样，开发集词表覆盖率 89.4%，样本分布不能代表完整 VQA v2。全局词表有 1,000 类，但抽样训练集只出现 829 类；词表内开发题有 81 道题的答案类别未在抽样训练集中出现。HTML 的 18 个案例来自真实开发集预测，置信分数使用独立 200 张图片拟合的温度缩放；它不是对所有新问题都可靠的概率保证。

### 下一次改进方向

1. 改进图片抽样，让训练集覆盖更多 Top-1,000 类，并在未按答案频次筛选的新开发图片或官方 Test-Dev 上评价。扩展为官方测试时使用官方评测工具。
2. 重点检查数量和其他类的错误。尝试让问题关注 CLIP 的局部图像特征，和当前全局向量分类头在同一数据划分上比较。
3. 使用多个随机种子和按图片重采样的配对置信区间，再判断 4.6 个百分点的差异是否稳定。若要回答用户上传的任意图片和问题，需增加推理服务并对词表外问题给出明确提示。

## 历史四题型实验

原本地测试集来自官方 val，含 **317 张图、320 条 QA**。图片、题目、标注、原预测和指标存档于本地 `data/archive/pilot_test/`，供复查旧实验。下列命令仅用于复现历史四题型 Notebook、演示和指标。

历史 pilot 用种子 42，从官方 train 分别抽取是非、数量、颜色、动物类别各 800 题。官方 val 按图片 ID 划分后，开发集和测试集各抽每类 80 题。按类型构建 2、11、12、10 类答案词表。训练跳过词表外目标，开发和测试把它们计为错误。

在可使用 GPU 的 Python 环境中，从本目录执行：

```powershell
python -m pip install -r requirements.txt
python scripts/download_sources.py
python scripts/prepare_data.py --sampling pilot --download-images
python scripts/build_notebook.py --execute
python scripts/export_public_demo.py
```

`build_notebook.py --execute` 会生成并逐单元执行 `notebooks/VQA实验.ipynb`，保留实际控制台输出和图表，同时生成 `results/summary.json`、`results/predictions.json`、`results/training_curves.png` 和本地离线版 `demo/demo.html`。也可以用 Jupyter 打开 Notebook，从头运行所有单元。最后一步将 HTML 改为公开仓库版，保留相同的真实预测，改从 COCO 官方地址加载图片。直接打开仓库版 HTML 时需要网络。

`prepare_data.py` 首次下载约 3700 张 COCO 图片，模型权重约 338 MiB；扩大或缩小 pilot 时修改 Notebook 第一段配置，以及执行 `prepare_data.py` 时的对应参数。图片下载使用 COCO 官方 HTTP 地址，因为本机访问该站 HTTPS 时证书握手失败；这些图片是公开数据，不传输个人信息。

### 历史评估口径

官方 train 与 val 的图片不交叉，开发集与测试集也按图片 ID 隔离。仅文本和冻结图文模型使用相同分类头、相同数据、相同轮次；冻结延续组与视觉末层微调组从同一 10 轮图文模型继续训练 2 轮。模型选择只看开发集，测试集用于最终报告。

Notebook 输出总体和分问题类型的分类准确率、答案词表覆盖率、训练曲线、参数量、估算 MACs 与单样本前向耗时。MACs 只包含卷积、线性层和注意力矩阵乘法；耗时不含读取图片和预处理。HTML 中的置信分数是未经校准的 softmax 输出。

### 历史运行结果

运行设备为 NVIDIA GeForce RTX 4060 Laptop GPU。测试集包含 320 题，答案词表覆盖率为 87.8%。下表是包含 OOV 题的分类 Exact Match Accuracy：

| 模型 | 开发集准确率 | 测试集准确率 |
| --- | ---: | ---: |
| 按问题类型猜训练集多数答案 | — | 23.4% |
| 仅文本 | 34.4% | 29.1% |
| 冻结图文融合 | 48.8% | 43.4% |
| 冻结图文融合再训练 2 轮 | 46.6% | 44.7% |
| 解冻视觉末层再训练 2 轮 | 45.9% | 43.4% |

HTML 展示模型按开发集结果选择为“冻结图文融合”。它在测试集的动物类别题达到 61.3%，颜色题 30.0%，数量题 27.5%，是非题 55.0%。本次局部微调没有带来总体增益，提示小样本下可能过拟合；请结合 Notebook 曲线和错误案例分析，不要只看总准确率。图文模型总参数约 151.8M，估算前向 7.39G MACs，单样本前向耗时中位数 33.2 ms（不含图片读取和预处理）。

课程要求的最终英文 IEEE 论文和 2–5 分钟演示视频可在本 pilot 结果确认后制作；本文档与 HTML 不代替论文。
