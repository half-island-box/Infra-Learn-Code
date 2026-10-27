# Model Code

这是一个用于学习大模型基础结构的个人学习项目，主要涵盖以下三方面：

- **模型结构**：从零实现 Transformer 相关的核心层与完整模型（LLaMA、Qwen3）。
- **训练框架**：实现朴素的数据并行（DP）、张量并行（TP）、流水线并行（PP，包括 GPipe 的 afab 模式与 1F1B 模式），以及对应的预训练流程。
- **推理框架**：暂未实现，后续逐步补充

## 目录结构

```text
.
├── pyproject.toml                  # 项目配置（uv 管理）
├── README.md
└── src/
    ├── data/
    │   └── sample.jsonl            # 样例训练数据
    ├── model_code/                 # 核心代码包
    │   ├── layers/                 # 基础层实现
    │   │   ├── Attention.py        # 多头注意力
    │   │   ├── RoPE.py             # 旋转位置编码
    │   │   ├── RMSNorm.py          # RMS 归一化
    │   │   ├── LayerNorm.py        # 层归一化
    │   │   ├── BatchNorm.py        # 批归一化
    │   │   └── Activation.py       # 激活函数
    │   ├── models/                 # 模型结构
    │   │   ├── llama.py            # LLaMA 模型
    │   │   └── qwen3.py            # Qwen3 模型
    │   ├── parallel/               # 并行训练框架
    │   │   ├── process_group_manager.py  # 进程组管理（DP/TP/PP）
    │   │   ├── tensor_parallel.py        # 张量并行（TP）
    │   │   ├── data_parallel.py          # 数据并行（DP，含梯度桶同步）
    │   │   └── pipeline_parallel.py      # 流水线并行（PP：afab / 1F1B）
    │   ├── data/                   # 数据加载与数据集
    │   │   └── pretrain_dataset.py # 预训练数据集
    │   ├── train/                  # 训练入口
    │   │   └── train.py            # 预训练主流程
    │   ├── dataloader.py           # 数据加载器
    │   └── utils.py                # 通用工具
    └── outputs/                    # 训练输出
        └── pretrain/
            └── epoch_0.pt          # 训练 checkpoint
```

## 声明

本仓库仅用于学习模型结构、训练框架与推理框架的基础原理，代码**不保证完全准确**，可能存在简化、未充分测试或有误之处，请勿直接用于生产环境。

---

## 学习代码来源

1. **推理框架和模型结构**
   - 视频教程：[B 站](https://space.bilibili.com/1223644844/lists/8717317?type=season)
   - 代码仓库：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

2. **训练框架**
   - 代码仓库：[huggingface/picotron](https://github.com/huggingface/picotron)
