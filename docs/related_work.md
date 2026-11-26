# 参考文献清单

按用途分组。每条注明它在本次实验里支撑哪一句结论，避免写报告时堆砌。

## 一、数据集与评测协议

| 论文 | 会议 | 用途 |
| --- | --- | --- |
| Antol et al., *VQA: Visual Question Answering* | ICCV 2015 | 任务定义；每题 10 个独立标注者；软准确率 `min(#humans that said it / 3, 1)` |
| Goyal et al., *Making the V in VQA Matter* | CVPR 2017 | VQA v2 数据集本体；配对平衡设计；v1 中"Do you see a…"一律答 yes 可达 87% |
| Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* | ICML 2021 | linear probe 协议出处；CLIP 自述在计数、细粒度识别上较弱 |

## 二、冻结主干 vs 微调主干

| 论文 | 会议 | 用途 |
| --- | --- | --- |
| Shen et al., *How Much Can CLIP Benefit Vision-and-Language Tasks?* | ICLR 2022 | CLIP-ViT 检测定位 AP 0.03，ImageNet-Res50 为 3.14；无 V&L 预训练时冻结视觉编码器优于解冻 |
| Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* | ICLR 2022 Oral | 随机初始化头 + 立即解冻 → 巨大梯度破坏预训练特征；解法 LP-FT（先冻后解） |

对应我们的结果：`finetune` 组比 `fusion` 低 0.8 个百分点，可归因于第二个机制。改造路径是 LP-FT：拿已训好的 fusion 头再解冻续训。

## 三、融合方式

`[q, v, q⊙v, |q−v|]` 属于特征级早期融合里的逐元素乘积系列。

| 论文 | 会议 | 用途 |
| --- | --- | --- |
| Fukui et al., *Multimodal Compact Bilinear Pooling for VQA and Visual Grounding* | EMNLP 2016 | 双线性融合起点 |
| Kim et al., *Hadamard Product for Low-rank Bilinear Pooling (MLB)* | ICLR 2017 | `q⊙v` 的直接出处，本实验最贴的一篇 |
| Ben-younes et al., *MUTAN: Multimodal Tucker Fusion for VQA* | CVPR 2017 | 加低秩约束的版本 |

## 四、闭集分类范式的局限

| 论文 | 出处 | 用途 |
| --- | --- | --- |
| Zhang et al., *When Open-Vocabulary Visual Question Answering Meets Causal Adapter: Benchmark and Approach* | AAAI 2025, 39(9): 9887-9895 | 现有 VQA 基准均为闭集，罕见答案无法输出，提出 OVVQA 基准 |
| Kervadec et al., *Estimating Semantic Structure for the VQA Answer Space* | arXiv:2006.05726, 2020 | 互斥 softmax 不反映答案间语义距离；提出 semantic loss |
| Ilievski et al., *A Simple Loss Function for Improving the Convergence and Accuracy of VQA Models* | arXiv:1708.00584 | 训练用众数硬标签、评测用 10 票软分，两者错配；改软交叉熵后 +1.2~1.6% |

最后一条最可直接落地：只需替换损失函数，特征缓存完全复用。

## 五、生成式范式对照

BLIP-2（arXiv:2301.12597）、LLaVA（arXiv:2304.08485）。把 VQA 当生成任务，不预设答案集合。作为"范式已换代"的对照写一段即可，不必展开。

---

## BibTeX

```bibtex
@inproceedings{antol2015vqa,
  title     = {VQA: Visual Question Answering},
  author    = {Antol, Stanislaw and Agrawal, Aishwarya and Lu, Jiasen and Mitchell, Margaret and Batra, Dhruv and Zitnick, C. Lawrence and Parikh, Devi},
  booktitle = {ICCV},
  year      = {2015}
}

@inproceedings{goyal2017making,
  title     = {Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering},
  author    = {Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
  booktitle = {CVPR},
  year      = {2017}
}

@inproceedings{radford2021learning,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {ICML},
  year      = {2021}
}

@inproceedings{shen2022much,
  title     = {How Much Can {CLIP} Benefit Vision-and-Language Tasks?},
  author    = {Shen, Sheng and Li, Liunian Harold and Tan, Hao and Bansal, Mohit and Rohrbach, Anna and Chang, Kai-Wei and Yao, Zhewei and Keutzer, Kurt},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{kumar2022finetuning,
  title     = {Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution},
  author    = {Kumar, Ananya and Raghunathan, Aditi and Jones, Robbie and Ma, Tengyu and Liang, Percy},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{fukui2016multimodal,
  title     = {Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding},
  author    = {Fukui, Akira and Park, Dong Huk and Yang, Daylen and Rohrbach, Anna and Darrell, Trevor and Rohrbach, Marcus},
  booktitle = {EMNLP},
  year      = {2016}
}

@inproceedings{kim2017hadamard,
  title     = {Hadamard Product for Low-rank Bilinear Pooling},
  author    = {Kim, Jin-Hwa and On, Kyoung-Woon and Lim, Woosang and Kim, Jeonghee and Ha, Jung-Woo and Zhang, Byoung-Tak},
  booktitle = {ICLR},
  year      = {2017}
}

@inproceedings{benyounes2017mutan,
  title     = {{MUTAN}: Multimodal Tucker Fusion for Visual Question Answering},
  author    = {Ben-younes, Hedi and Cadene, Remi and Cord, Matthieu and Thome, Nicolas},
  booktitle = {CVPR},
  year      = {2017}
}

@inproceedings{zhang2025ovvqa,
  title     = {When Open-Vocabulary Visual Question Answering Meets Causal Adapter: Benchmark and Approach},
  author    = {Zhang, Feifei and Zhang, Zhaoyi and Zhang, Xi and Xu, Changsheng},
  booktitle = {AAAI},
  volume    = {39},
  number    = {9},
  pages     = {9887--9895},
  year      = {2025}
}

@article{kervadec2020estimating,
  title   = {Estimating Semantic Structure for the {VQA} Answer Space},
  author  = {Kervadec, Corentin and Antipov, Grigory and Baccouche, Moez and Wolf, Christian},
  journal = {arXiv:2006.05726},
  year    = {2020}
}

@article{ilievski2017simple,
  title   = {A Simple Loss Function for Improving the Convergence and Accuracy of Visual Question Answering Models},
  author  = {Ilievski, Ilija and Feng, Jiashi},
  journal = {arXiv:1708.00584},
  year    = {2017}
}
```

## 数量建议

必引 5 篇（第一、二节）是底线。融合挑 1 篇（MLB 最贴）。闭集局限挑 1 篇。总计 7 到 8 篇对大作业的 Related Work 刚好，再多会显得堆砌。生成式那两篇只在 Discussion 提一句，不进 Related Work。
