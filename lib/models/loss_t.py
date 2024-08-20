import torch
import copy
import torch.nn.functional as F
def get_neg_scores(scores, scores_masked):
    bsz = len(scores)
    batch_indices = torch.arange(bsz).to(scores.device)
    _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

    sample_min_idx = 1  # 跳过正样本
    use_hard_negative = True  # 总是使用硬负样本挖掘
    hard_pool_size = 20  # 硬负样本池大小

    if use_hard_negative:
        sample_max_idx = min(sample_min_idx + hard_pool_size, bsz)
    else:
        sample_max_idx = bsz

    # 随机选择一个在硬负样本范围内的负样本
    sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx, size=(bsz,)).to(scores.device)]
    sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N,)
    return sampled_neg_scores

def get_frame_trip_loss(query_context_scores):
    bsz = len(query_context_scores)
    diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
    pos_scores = query_context_scores[diagonal_indices, diagonal_indices]
    # query_context_scores_masked = copy.deepcopy(query_context_scores)
    query_context_scores_masked = query_context_scores.clone().detach()
    query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
    pos_query_neg_context_scores = get_neg_scores(query_context_scores, query_context_scores_masked)
    neg_query_pos_context_scores = get_neg_scores(query_context_scores.transpose(0, 1), query_context_scores_masked.transpose(0, 1))
    loss_neg_ctx = get_ranking_loss(pos_scores, pos_query_neg_context_scores)
    loss_neg_q = get_ranking_loss(pos_scores, neg_query_pos_context_scores)
    return loss_neg_ctx + loss_neg_q

def get_ranking_loss(pos_score, neg_score):
    return torch.clamp(0.2 + neg_score - pos_score, min=0).sum() / len(pos_score)


import torch
import torch.nn as nn


# 定义计算MSE损失的函数
def calculate_mse_loss(input_tensor, target_tensor):
    """
    计算两个张量之间的均方误差损失。

    参数:
    - input_tensor (torch.Tensor): 输入张量。
    - target_tensor (torch.Tensor): 目标张量。

    返回:
    - float: 计算出的均方误差损失。
    """
    # 创建损失函数实例
    mse_loss = nn.MSELoss()
    # 使用PyTorch的Softmax和交叉熵损失
    A_prob_torch = F.softmax(input_tensor, dim=1)
    B_prob_torch = F.softmax(target_tensor, dim=1)
    # 计算损失
    # loss = mse_loss(input_tensor, target_tensor)
    loss = F.binary_cross_entropy(A_prob_torch,  B_prob_torch)
    return loss

#
# # 示例使用
# # 假设 sims2 和 sims_gt 是已经定义好的两个张量
# input_tensor = torch.randn(32, 32, dtype=torch.float32)
# target_tensor= torch.randn(32, 32, dtype=torch.float32)

