import json
import os
import torch
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练用的简单 JSONL 数据集。
    每一行是一个 JSON 对象，至少包含字段 ``text``（或 ``content``）。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"data_path not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = str(sample.get("text", sample.get("content", "")))

        # 先编码正文（留出 BOS / EOS 两个位置）
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        # 加上特殊符
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        pad_len = self.max_length - len(tokens)
        if pad_len > 0:
            input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
        else:
            input_ids = tokens[: self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 填充部分写为 -100，CrossEntropyLoss 默认会忽略
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 告知 attention 不需要计算 pad 位置
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids, attn_mask, labels
