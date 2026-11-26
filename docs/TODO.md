# 待办

做完这五项即满足大作业要求。

## 1. Params 总表

现在只报了分类头 1.56M 与解冻部分 7.48M，缺整模型总计。

- [ ] 统计 CLIP 图像编码器 + 文本编码器 + 分类头的总参数量
- [ ] 三组各报可训练参数量与占比
- [ ] 写进 Table 4

## 2. MACs / FLOPs

- [ ] 移植 `scripts/experiment_source.py:425` 的 `clip_macs()` 解析式计数
- [ ] 报告注明口径：只算卷积、线性层、注意力矩阵乘，不含 LayerNorm、激活、预处理、磁盘 IO；图像 224×224、文本 77 token
- [ ] 写进 Table 4

## 3. 推理耗时 / 显存

- [ ] 单样本前向延迟中位数：预热 5 次、测量 30 次
- [ ] 显存峰值：`torch.cuda.max_memory_allocated()`
- [ ] 在 RTX 4060 上跑三组
- [ ] 写进 Table 4

> 第 1–3 项由同一个脚本 `scripts/benchmark_cost.py` 一次产出，输出 `results/cost_table.json`。

## 4. 演示录屏或截图

- [ ] `demo/three_way.html` 录 30–60 秒，展示同一道题三组答案与置信度的差异
- [ ] 抽 3–5 张截图进论文 Figure

## 5. 英文 IEEE 论文

- [ ] 确认模板版本与页数限制
- [ ] 骨架：Introduction / Related Work / Method / Experiments / Results and Discussion / Conclusion and Limitations
- [ ] 写作顺序：Method → Results → Experiments → Related Work → Introduction → Conclusion
- [ ] 参考文献用 `docs/related_work.md` 的 BibTeX
- [ ] 编译出 PDF

## 完成标准

五项全部勾掉，且 Table 4 的 Params / MACs / 延迟 / 显存四项在三组上都有数。
