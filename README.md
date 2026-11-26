# VQA v2 图文视觉问答课程实验

当前版本在自建的 Top‑1,000 闭集数据上实际训练了仅文本与冻结 CLIP 图文模型，提供 [已运行的 Top‑1,000 Notebook](notebooks/v3-20989-2-main.ipynb) 和 [HTML 问答案例](demo/top1000.html)。三组对齐对照见 [三组 Notebook](notebooks/v3-20989-3-aligned.ipynb)。主结果基于 **20,989 条**训练 QA；12,400 条与 3,200 条四题型两个规模作为历史对照保留，跨规模不能比绝对分数，原因见 [notebooks/README.md](notebooks/README.md)。

Notebook 按 `v<阶段>-<训练 QA 数>-<对照组数>-<实验类型>.ipynb` 命名，三个阶段的规模与成绩对照见 [notebooks/README.md](notebooks/README.md)。原始数据、模型权重和图片存放在本地。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 数据下载和预处理、实验源码、Notebook 与演示生成脚本、数据自检与显著性检验 |
| `notebooks/` | 各数据规模的实验 Notebook，命名见该目录 README |
| `demo/` | HTML 演示和模板 |
| `results/` | 预测、指标和图表 |
| `docs/` | [参考文献清单与 BibTeX](docs/related_work.md)、[待办事项](docs/TODO.md) |
| `data/`、`vendor/` | 本地数据、模型权重与第三方实现，按需生成 |

## 数据与模型

- 问题、人工答案： [VQA v2 官方下载页](https://visualqa.org/download)，使用 real images 的 train/val 标注和问题文件。
- 图片： [COCO 2014](https://cocodataset.org/)，只下载清单中引用的图片。COCO [使用条款](https://cocodataset.org/dataset/termsofuse.htm)说明图片版权属于原上传者；本仓库的 HTML 引用官方图片地址，不附带图片副本。
- 预训练模型与分词实现： [OpenAI CLIP](https://github.com/openai/CLIP)，ViT-B/32。权重 SHA256 在代码中验证；运行 `scripts/download_sources.py` 后第三方实现会位于 `vendor/CLIP-main`，并保留原许可。

## 数据冻结与自检

三份清单在实验结果定稿后冻结，不再改动。`scripts/verify_data.py` 复算全部统计并比对 sha256，任何一项不一致就退出并报错：

```powershell
python scripts/verify_data.py
```

| 清单 | QA 数 | 图片数 | sha256 前 16 位 |
| --- | ---: | ---: | --- |
| `data/manifest_top1000.json`（train） | 20,989 | 2,000 | `5836493a7ee85bd2` |
| `data/manifest_top1000_dev.json`（dev） | 8,083 | 800 | `cb7c523f9ee3843a` |
| `data/manifest_top1000_test.json`（test） | 8,072 | 800 | `caf699f84ce7abbd` |

测试清单不入库，用 `python scripts/prepare_data.py --sampling top1000-test` 重新生成即可复现同一份（种子 42 固定）。

## Top‑1,000 扩容数据集

按完整官方训练集的 `multiple_choice_answer` 频次建立全局 Top‑1,000 词表，同频答案按字母顺序排序。对官方训练集的全部题型先过滤词表外的 QA，再按图片统计保留的 QA 数。每张候选图至少有 6 条高频 QA。

抽样按**密度分层等比例**进行：所有保留 QA 数不少于 6 的图片都进入候选池，按保留 QA 数分层，每层抽取相同比例，名额用最大余数法补齐。这样得到的子集保留官方自然的每图 QA 分布，不再把密度人为压在 6–7 题。入选图片的全部词表内 QA 都会被保留。

训练候选池为 **17,297 张图 / 183,589 条 QA**，按 11.56% 抽取 **2,000 张图、20,989 条 QA**，每图 QA 数覆盖 6–78、均值 10.49。开发集沿用同一词表和答案 ID，在官方 val 的开发图片池内以 19.04% 抽取 **800 张图、8,083 条 QA**（密度 6–45、均值 10.10）；测试集在另一批 val 图片池内以 19.15% 抽取 **800 张图、8,072 条 QA**（密度 6–45、均值 10.09）。开发与测试图片池按图片 ID 以种子 42 对半切分，互不重叠。

默认抽样种子为 42。训练清单保存在 `data/manifest_top1000.json`，开发清单保存在 `data/manifest_top1000_dev.json`，测试清单保存在 `data/manifest_top1000_test.json`，前两者引用 `data/images/train2014/` 与 `data/images/val2014/` 中的图片，后者引用 `data/images/val2014/`。先生成训练清单，再生成开发与测试清单。只生成清单可省略 `--download-images`。

```powershell
python -m pip install -r requirements.txt
python scripts/download_sources.py
python scripts/prepare_data.py --sampling top1000 --download-images
python scripts/prepare_data.py --sampling top1000-dev --download-images
python scripts/prepare_data.py --sampling top1000-test --download-images
python scripts/build_notebook.py --source scripts/top1000_experiment_source.py --output notebooks/v3-20989-2-main.ipynb --execute
python scripts/export_top1000_demo.py
```

默认执行 `python scripts/prepare_data.py` 会生成 Top‑1,000 训练清单。Notebook 使用全局 1,000 类分类头，选出开发集表现较好的模型，生成 `results/top1000_summary.json`、`results/top1000_predictions.json` 和两张图。训练后的分类头保存在本地 `results/top1000_head.pt`，不提交 Git。只生成 Notebook 结构可省略 `--execute`，但其中不会出现真实结果。仓库版 HTML 引用 COCO 官方图片地址，打开时需要网络。

默认使用本仓库的 `data/` 和 `vendor/`。如需复用同一台电脑上的已有公开数据与 CLIP 实现，可设置 `VQA_DATA_ROOT` 和 `VQA_VENDOR_ROOT` 指向相应目录；未设置时，以上命令独立运行。

## 调参和官方评测

训练集、开发集与测试集按官方 train/val 图片隔离，开发与测试图片池互不重叠。开发集 800 张图按图片 ID 再分成 600 张模型选择图片和 200 张置信度校准图片；两者合计 **8,083 条词表内 QA**。同一批图片共有 **8,969 条**未按答案过滤的 QA，词表覆盖率 90.1%。测试集 800 张图含 **8,072 条词表内 QA**、8,919 条未过滤 QA，覆盖率 90.5%。题型标注只用于结果分组，不用来限制模型输出。训练词表只来自官方 train 标注。

Top-1,000 训练子集只包含词表内题；未过滤开发集保留全部 886 道词表外题。在 Exact Match 中，这些题只有模型预测到其标注答案才能计为正确，而封闭词表模型无法输出该答案，因此计为错误。VQA-style 评分仍逐题与 10 个人工答案比较。

Notebook 同时报告词表内与未过滤开发题的 Exact Match，并依据 10 个人工答案计算带短答案规范化的 **VQA-style** 分数。这是本地实现的开发集分析指标；正式 [VQA Accuracy](https://visualqa.org/evaluation.html) 应以官方评测结果为准。当前所有新结果均为开发集成绩，不能称为 VQA v2 官方测试成绩。模型定稿后，可按 [官方结果格式](https://visualqa.org/evaluation.html) 导出 `question_id`、`answer` JSON 并提交 Test-Dev。

### 本次 Top-1,000 真实运行结果

设备：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB），CUDA 12.6。分类头训练 16 轮，按 600 张开发图片上的 6,128 道词表内题选最佳轮次与模型。另有 200 张图片、1,955 道词表内题仅用于温度缩放。下表的词表内成绩基于全部 8,083 道开发题，未过滤成绩基于同一批图片的全部 8,969 道题。

| 模型 | 词表内 Exact Match | 未过滤 Exact Match | 未过滤 VQA-style |
| --- | ---: | ---: | ---: |
| 训练集多数答案 `no` | — | 21.2% | 26.6% |
| 仅文本（最佳第 9 轮） | 36.8% | 33.1% | 40.9% |
| 冻结 CLIP 图文融合（最佳第 16 轮） | **42.6%** | **38.4%** | **46.6%** |

未过滤开发题的 Exact Match 按官方 `answer_type` 分组：

| 答案类型 | 题数 | 仅文本 | 图文融合 |
| --- | ---: | ---: | ---: |
| 是非 | 3,320 | 57.1% | 61.4% |
| 数量 | 1,257 | 27.1% | 27.1% |
| 其他 | 4,392 | 16.7% | 24.1% |

图文模型在未过滤开发题上比仅文本高 5.3 个百分点（33.1% → 38.4%），增益几乎全部来自其他类（+7.4）。数量题在这组 16 轮设置下零增益，与三组对齐实验的 +4.7 不一致；差异来自轮次预算与选模轮次不同，说明数字类答案的收益对训练长度敏感，不宜单点引用。

开发集词表覆盖率 90.1%，样本分布不能代表完整 VQA v2。全局词表有 1,000 类，训练集只出现 900 类；词表内开发题有 60 道题的答案类别未在训练集中出现。HTML 的 18 个案例来自真实开发集预测，置信分数使用独立 200 张图片拟合的温度缩放；它不是对所有新问题都可靠的概率保证。

### 下一次改进方向

1. 改进图片抽样，让训练集覆盖更多 Top-1,000 类，并在未按答案频次筛选的新开发图片或官方 Test-Dev 上评价。扩展为官方测试时使用官方评测工具。
2. 重点检查数量和其他类的错误。尝试让问题关注 CLIP 的局部图像特征，和当前全局向量分类头在同一数据划分上比较。
3. 用 `scripts/significance_test.py` 的配对检验复核 5.7 个百分点的图像增益，再补跑其他随机种子确认排序稳定。若要回答用户上传的任意图片和问题，需增加推理服务并对词表外问题给出明确提示。

## 三组对齐对照实验

`notebooks/v3-20989-3-aligned.ipynb` 在同一份 Top-1,000 数据上对比三种训练方案：

| 组 | 图像输入 | 视觉编码器 | 分类头 |
| --- | --- | --- | --- |
| `text` | 无 | 文本编码器冻结 | 随机初始化后训练 |
| `fusion` | 有 | 视觉编码器**冻结** | 随机初始化后训练 |
| `finetune` | 有 | 视觉编码器**末层解冻** | 随机初始化后训练 |

三组都从随机初始化的分类头开始，训练相同轮次，数据划分、随机种子、优化器、学习率与 batch size 完全一致。**唯一差异是"是否使用图像"和"是否微调视觉编码器"**，因此差异可以直接归因，不必再引入续训轮次、学习率等额外变量。

这与历史 pilot 的设计不同：历史版本让冻结组与微调组从"已训好的冻结模型"续训 2 轮，轮次与超参不一致，无法干净归因。

```powershell
python scripts/build_notebook.py --source scripts/three_way_experiment_source.py --output notebooks/v3-20989-3-aligned.ipynb --execute
python scripts/export_three_way_demo.py
```

微调组在训练时实时编码图片（梯度需流过视觉编码器），因此比两组缓存特征的方案慢得多。结果写入 `results/three_way_summary.json`、`results/three_way_predictions.json`、`results/three_way_training_curves.png` 和 `results/three_way_by_type.png`；`demo/three_way.html` 展示同一道题上三组各自的答案与置信分数。

训练完成后会把三组权重与训练历史保存到本地 `results/three_way_checkpoint.pt`（不提交 Git）。设置环境变量 `THREE_WAY_REUSE=1` 可跳过训练、只重跑评估与出图环节，便于在改动评分口径时快速迭代：

```powershell
$env:THREE_WAY_REUSE = "1"
python scripts/three_way_experiment_source.py
```

### 三组真实运行结果

设备：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB），CUDA 12.6，PyTorch 2.14。三组均训练 10 轮，batch 32，分类头学习率 8e-4，微调组的视觉编码器学习率 1e-5。训练集 2,000 张图、20,989 条词表内 QA。按 600 张选模图片上的 6,061 道词表内题选最佳轮次与模型；另有 200 张图片、2,022 道词表内题只用于温度缩放。

| 模型 | 词表内 Exact Match | 未过滤 Exact Match | 未过滤 VQA-style | 最佳轮次 |
| --- | ---: | ---: | ---: | ---: |
| 训练集多数答案 `no` | — | 21.2% | 26.6% | — |
| 仅文本（冻结文本编码器） | 35.9% | 32.3% | 40.4% | 4 |
| 图文融合（冻结视觉编码器） | **41.6%** | **37.5%** | **45.4%** | 10 |
| 图文融合 + 微调视觉末层 | 40.8% | 36.8% | 44.7% | 8 |

未过滤开发题（8,969 条）按官方 `answer_type` 分组：

| 答案类型 | 题数 | 仅文本 | 图文融合 | 图文融合 + 微调 |
| --- | ---: | ---: | ---: | ---: |
| 是非 | 3,320 | 56.8% | 60.7% | 60.2% |
| 数量 | 1,257 | 25.2% | 29.9% | 28.0% |
| 其他 | 4,392 | 15.8% | 22.2% | 21.6% |

三组训练预算完全对齐，结论可以直接归因：**使用图像带来 +5.7 个百分点的词表内增益（35.9% → 41.6%），在冻结视觉编码器之上再解冻末层反而低 0.8 个百分点**（41.6% → 40.8%）。图像增益按题型递减：其他类 +6.4、数量类 +4.7、是非类 +3.9，开放类答案从视觉信息中获益最多，与 CLIP 的图文匹配预训练目标一致。

图文融合组的最佳轮次落在第 10 轮，选模准确率在末轮仍为最高，说明 10 轮预算对该组偏紧。微调组在第 8 轮见顶后连续下滑，末轮训练损失 0.42，而融合组为 0.72；可训练参数从 156 万增加到 904 万，过拟合发生得更早也更陡。

与旧版 12,400 条训练集相比，三组分别提升 1.4、2.8、2.9 个百分点，图像带来的增益从 4.3 扩大到 5.7。新增样本的作用集中在视觉侧：仅文本组涨幅最小，说明扩样本主要补充了图像与答案之间的对应关系，问题措辞多样性的贡献有限。

微调组原本每轮要重复解码约两万次 JPEG 并重算 12 个 Transformer block，单轮耗时 571 秒。由于只有最后一个 block、`ln_post` 与 `proj` 参与更新，前 11 个 block 的输出在每轮上完全相同，`build_trunk_cache` 把它们预计算并缓存，训练时每批只剩一个 block 的前向与反向。拆分前向与整模型前向的最大绝对误差为 `0.00e+00`，单轮降到 23 秒。

两类误差仍然明显：数量题 29.9%、其他类 22.2%，都远低于是非题的 60.7%。其他类占开发题的 49.0%，是拉低总体分数的主因。HTML 展示 21 个真实开发集案例，其中包含"仅文本答对但图文融合答错"的题，用于说明图像输入并非在所有问题上都有帮助。

### 差异显著性

`scripts/significance_test.py` 读取 `results/three_way_predictions.json`，对三组两两做配对检验：

```powershell
python scripts/significance_test.py
```

同时给出 McNemar 检验（只看两组判断不一致的题）与按图片整块重采样的 bootstrap 95% 置信区间（同一张图的多道题不独立，逐题重采样会低估方差）。结果写入 `results/significance.json`。

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

`build_notebook.py --execute` 会生成并逐单元执行 `notebooks/v1-3200-4-pilot.ipynb`，保留实际控制台输出和图表，同时生成 `results/summary.json`、`results/predictions.json`、`results/training_curves.png` 和本地离线版 `demo/demo.html`。也可以用 Jupyter 打开 Notebook，从头运行所有单元。最后一步将 HTML 改为公开仓库版，保留相同的真实预测，改从 COCO 官方地址加载图片。直接打开仓库版 HTML 时需要网络。

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
