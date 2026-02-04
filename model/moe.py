import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
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
                 moe_args: MoeArgs):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate_img = gate_img
        self.gate_txt = gate_txt
        self.task_gate = task_gate
        self.args = moe_args
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 平衡 input vs task gate

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

        # Weight normalization
        weights = F.softmax(weights.float(), dim=-1).to(inputs.dtype)

        # Dispatch
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            idx_1, idx_2, nth = torch.where(selected_experts == i)
            if len(idx_1) > 0:
                expert_out = expert(inputs[idx_1, idx_2])
                w = weights[idx_1, idx_2, nth].unsqueeze(-1)
                results[idx_1, idx_2] += w * expert_out

        return results, l_aux.float()


# class MoeLayer(nn.Module):
#     def __init__(self, experts: List[nn.Module], input_gate: nn.Module, task_gate: nn.Module, moe_args: MoeArgs):
#         super().__init__()
#         assert len(experts) > 0
#         self.experts = nn.ModuleList(experts)
#         self.input_gate = input_gate
#         self.task_gate = task_gate
#         self.args = moe_args
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, inputs: torch.Tensor, task_param) -> torch.Tensor:
#         input_gate_logits = self.input_gate(inputs)
#         task_gate_logits = self.task_gate(task_param)
#
#         gate_logits = (1 - self.alpha) * input_gate_logits + self.alpha * task_gate_logits
#
#         # gate_logits = input_gate_logits
#
#         weights, selected_experts = torch.topk(
#             gate_logits, self.args.num_experts_per_tok
#         )
#
#         # calculate aux_loss
#         weights_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(inputs.dtype)
#         average_weight = torch.mean(weights_softmax, dim=[0, 1])
#
#         # use top 2 to cal
#         indices_top2 = F.one_hot(selected_experts, num_classes=self.args.num_experts).sum(dim=2)
#         average_count = torch.mean(indices_top2.float(), dim=[0, 1]).to(inputs.dtype)
#
#         # cal aux loss, Load-Balancing Loss
#         l_aux = torch.mean(average_weight * average_count) * self.args.num_experts
#
#         weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
#
#         results = torch.zeros_like(inputs)
#
#         for i, expert in enumerate(self.experts):
#             idx_1, idx_2, nth_expert = torch.where(selected_experts == i)
#             results[idx_1, idx_2] += weights[idx_1, idx_2, nth_expert, None] * expert(inputs[idx_1, idx_2])
#
#         return results, l_aux.float()
