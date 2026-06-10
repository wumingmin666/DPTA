import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS

@MODELS.register_module()
class PrototypeContrastiveLoss(nn.Module):
    """
    Contrastive Loss (L_con) for Metric Branch.
    Uses InfoNCE-like formulation to align query embeddings with class prototypes.
    """
    def __init__(self, temperature=0.07, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_feats, prototype_feats, gt_labels):
        """
        Args:
            query_feats (Tensor): [N, D] Embeddings from the query image.
            prototype_feats (Tensor): [C, D] Embeddings from the prototype library.
            gt_labels (Tensor): [N] The class indices for the queries.
        """
        if query_feats.size(0) == 0:
            return query_feats.sum() * 0

        # 1. 归一化特征 (Cosine Similarity 前置步骤)
        query_norm = F.normalize(query_feats, p=2, dim=1)
        proto_norm = F.normalize(prototype_feats, p=2, dim=1)

        # 2. 计算相似度矩阵 (Logits) [cite: 113]
        # logits shape: [N, C] (N个查询对 C个类别的相似度)
        logits = torch.matmul(query_norm, proto_norm.t()) / self.temperature

        # 3. 计算 Cross Entropy (等价于 InfoNCE) [cite: 129]
        # 目标是将 query 拉向 gt_labels 对应的 prototype
        loss = self.criterion(logits, gt_labels)

        return self.loss_weight * loss