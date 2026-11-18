# VQA v2 图文视觉问答课程实验

本仓库提供可重新运行的 `VQA实验.ipynb` 和由真实测试预测生成的 `demo.html`。实验将是非、数量、颜色、动物类别四种短答案问题建模为分类任务，比较多数类、仅文本、冻结 CLIP 图文融合，以及视觉编码器末层微调。仓库版 HTML 需联网读取 COCO 官方图片；训练数据、模型权重和原始图片不会提交到 Git。

## 数据与模型

- 问题、人工答案： [VQA v2 官方下载页](https://visualqa.org/download)，使用 real images 的 train/val 标注和问题文件。
- 图片： [COCO 2014](https://cocodataset.org/)，只下载清单中引用的图片。COCO [使用条款](https://cocodataset.org/dataset/termsofuse.htm)说明图片版权属于原上传者；本仓库的 HTML 引用官方图片地址，不附带图片副本。
- 预训练模型与分词实现： [OpenAI CLIP](https://github.com/openai/CLIP)，ViT-B/32。权重 SHA256 在代码中验证；运行 `download_sources.py` 后第三方实现会位于 `vendor/CLIP-main`，并保留原许可。

本 pilot 使用固定随机种子 42，从官方 train 按问题文本前缀筛选是非、数量、颜色、动物类别各 800 题；是非和动物类别还使用官方 `answer_type` 排除不符合该任务的题。官方 val 先按图片 ID 分成开发和测试图片池，再各取每类 80 题。按类型从训练子集建立 2、11、12、10 类答案词表，预测时只考虑当前类型的答案。词表外的训练题不参与分类头训练，词表外的开发/测试题保留并计为错误。准确率是分类 Exact Match，与官方 VQA Accuracy 不同；均衡抽样的 pilot 成绩也不能代表完整 VQA v2 基准成绩。若扩展为任意问题的实时问答，还需要在推理时判定问题类型。

## 运行

在可使用 GPU 的 Python 环境中，从本目录执行：

```powershell
python -m pip install -r requirements.txt
python download_sources.py
python prepare_data.py --download-images
python build_notebook.py --execute
python export_public_demo.py
```

第三步会生成并逐单元执行 `VQA实验.ipynb`，保留实际控制台输出和图表，同时生成 `results/summary.json`、`results/predictions.json`、`results/training_curves.png` 和本地离线版 `demo.html`。也可以用 Jupyter 打开 Notebook，从头运行所有单元。最后一步将 HTML 改为公开仓库版，保留相同的真实预测，改从 COCO 官方地址加载图片。直接打开仓库版 HTML 时需要网络。

`prepare_data.py` 首次下载约 3700 张 COCO 图片，模型权重约 338 MiB；扩大或缩小 pilot 时修改 Notebook 第一段配置，以及执行 `prepare_data.py` 时的对应参数。图片下载使用 COCO 官方 HTTP 地址，因为本机访问该站 HTTPS 时证书握手失败；这些图片是公开数据，不传输个人信息。

## 评估口径

官方 train 与 val 的图片不交叉，开发集与测试集也按图片 ID 隔离。仅文本和冻结图文模型使用相同分类头、相同数据、相同轮次；冻结延续组与视觉末层微调组从同一 10 轮图文模型继续训练 2 轮。模型选择只看开发集，测试集用于最终报告。

Notebook 输出总体和分问题类型的分类准确率、答案词表覆盖率、训练曲线、参数量、估算 MACs 与单样本前向耗时。MACs 只包含卷积、线性层和注意力矩阵乘法；耗时不含读取图片和预处理。HTML 中的置信分数是未经校准的 softmax 输出。

## 本次真实运行结果

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
