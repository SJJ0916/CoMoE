import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from tabulate import tabulate
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module],
                 gate_img: nn.Module,   # 图像专用门控
                 gate_txt: nn.Module,   # 文本专用门控
                 task_gate: nn.Module,  # 共享任务门控（可选）
                 moe_args: MoeArgs,
                 index):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate_img = gate_img
        self.gate_txt = gate_txt
        self.task_gate = task_gate
        self.args = moe_args
        self.index = index
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 平衡 input vs task gate

        self.register_buffer(
            "expert_count_accum",
            torch.zeros(moe_args.num_experts, dtype=torch.long)
        )
        self.register_buffer(
            "token_count_accum",
            torch.zeros(1, dtype=torch.long)
        )

    def forward(self, inputs: torch.Tensor, task_param: torch.Tensor, is_text: bool) -> tuple:
        """
        inputs: (L, B, D)
        task_param: (D,) or (B, D)
        is_text: bool —— 标识当前 batch 是文本还是图像
        """
        L, B, D = inputs.shape

        # 扩展 task_param
        if task_param.dim() == 1:
            task_param = task_param.unsqueeze(0).expand(B, -1)  # (B, E)
        task_logits = self.task_gate(task_param).unsqueeze(0)  # (1, B, E)

        # 选择门控
        if is_text:
            input_logits = self.gate_txt(inputs)  # (L, B, E)
        else:
            input_logits = self.gate_img(inputs)  # (L, B, E)

        # 融合
        gate_logits = (1 - self.alpha) * input_logits + self.alpha * task_logits  # (L, B, E)

        # Top-k routing
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok, dim=-1)
        # Auxiliary loss
        softmax_logits = F.softmax(gate_logits.float(), dim=-1)
        avg_weight = softmax_logits.mean(dim=[0, 1])  # (E,)
        expert_mask = F.one_hot(selected_experts, self.args.num_experts).sum(dim=2).float()
        avg_count = expert_mask.mean(dim=[0, 1])  # (E,)
        l_aux = torch.mean(avg_weight * avg_count) * self.args.num_experts

        weights = F.softmax(weights.float(), dim=-1).to(inputs.dtype)

        # ===== 打印专家选择分布（调试用）=====
        with torch.no_grad():
            # selected_experts: (L, B, K)
            # expert_counts = (
            #     F.one_hot(selected_experts, self.args.num_experts)
            #     .sum(dim=(0, 1, 2))  # (E,)
            #     .cpu()
            # )
            expert_counts = (
                F.one_hot(selected_experts, self.args.num_experts)
                .sum(dim=(0, 1, 2))  # (E,)
            )

            self.expert_count_accum += expert_counts
            self.token_count_accum += selected_experts.numel()

        # --- 新增结束 ---

        # Dispatch (保持原样)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            idx_1, idx_2, nth = torch.where(selected_experts == i)
            if len(idx_1) > 0:
                expert_out = expert(inputs[idx_1, idx_2])
                w = weights[idx_1, idx_2, nth].unsqueeze(-1)
                results[idx_1, idx_2] += w * expert_out

        return results, l_aux.float()