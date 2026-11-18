# VQA v2 图文视觉问答课程实验

当前 VQA v2 实验使用 Top‑1,000 训练集和开发集。用开发集调参、选模型，模型定稿后提交官方 `test-dev` 评测。仓库同时保留历史四题型的 [实验 Notebook](notebooks/VQA实验.ipynb) 和 [网页演示](demo/demo.html)。原始数据、模型权重和图片存放在本地。

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
python -m pip install requests Pillow
python scripts/download_sources.py --annotations-only
python scripts/prepare_data.py --sampling top1000 --download-images
python scripts/prepare_data.py --sampling top1000-dev --download-images
```

默认执行 `python scripts/prepare_data.py` 会生成 Top‑1,000 训练清单。Top‑1,000 子集使用全局分类标签，后续训练需要将分类头切换到这个词表。

## 调参和官方评测

使用训练集拟合模型，用 4,960 条开发 QA 调参、早停和选择模型。开发集结果反映 Top‑1,000 过滤后的高频答案子集。定稿后，从 [VQA v2 官网](https://visualqa.org/download.html) 获取官方测试题目及对应图片，按照 [官方结果格式](https://visualqa.org/evaluation.html) 将所需的每题预测写为含 `question_id` 和 `answer` 的 JSON 列表，再提交到 [EvalAI VQA Challenge](https://eval.ai/web/challenges/challenge-page/830/participate) 的 **Test-Dev** 阶段。官方评测使用 VQA Accuracy，开发集的分类 Exact Match 另外记录。

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
