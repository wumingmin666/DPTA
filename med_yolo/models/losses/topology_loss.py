import torch
import torch.nn as nn
from mmyolo.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps


def calculate_diou_matrix(bboxes1, bboxes2):
    """
    计算两组框之间的 DIoU 矩阵
    bboxes1: [N, 4]
    bboxes2: [M, 4]
    Returns: [N, M]
    """
    # 1. 基础 IoU [N, M]
    iou_matrix = bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False)
    
    # 2. 计算中心点
    # bboxes: [x1, y1, x2, y2]
    # center: [(x1+x2)/2, (y1+y2)/2]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    center1_x = bboxes1[:, 0] + w1 / 2
    center1_y = bboxes1[:, 1] + h1 / 2
    
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    center2_x = bboxes2[:, 0] + w2 / 2
    center2_y = bboxes2[:, 1] + h2 / 2
    
    # 3. 计算中心点距离平方 rho^2 [N, M]
    # 利用广播机制: [N, 1] - [1, M]
    d2 = (center1_x[:, None] - center2_x[None, :]) ** 2 + \
         (center1_y[:, None] - center2_y[None, :]) ** 2
         
    # 4. 计算最小闭包区域的对角线距离平方 c^2 [N, M]
    # 闭包左上角: min(x1_i, x1_j)
    lt_x = torch.min(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])
    lt_y = torch.min(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])
    
    # 闭包右下角: max(x2_i, x2_j)
    rb_x = torch.max(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    rb_y = torch.max(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])
    
    # 闭包宽高 (防止除0)
    cw = (rb_x - lt_x).clamp(min=1e-6)
    ch = (rb_y - lt_y).clamp(min=1e-6)
    
    c2 = cw ** 2 + ch ** 2
    
    # 5. DIoU = IoU - (rho^2 / c^2)
    diou_matrix = iou_matrix - (d2 / c2)
    
    return diou_matrix

@MODELS.register_module()
class TopologyEnergyLoss(nn.Module):
    """
    Anatomical Topology Constraints (L_energy) implementation.
    Calculates the MSE loss between the DIoU matrix of predicted boxes
    and the DIoU matrix of ground truth boxes.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred_bboxes, gt_bboxes, valid_mask=None):
        """
        Args:
            pred_bboxes (Tensor): [N, 4] Predicted bboxes (xyxy) for positive samples.
            gt_bboxes (Tensor): [N, 4] Corresponding GT bboxes.
            valid_mask (Tensor, optional): Mask for valid samples if needed.
        """
        # 如果当前批次正样本少于2个，无法构成拓扑图，损失为0
        N = pred_bboxes.size(0)
        if N < 2:
            return pred_bboxes.sum() * 0

        # 1. 计算 GT 的能量矩阵 (DIoU Matrix) [cite: 134]
        # mode='diou', is_aligned=False 会计算所有两两之间的 DIoU
        # energy_gt shape: [N, N]
        # energy_gt = bbox_overlaps(gt_bboxes, gt_bboxes, mode='diou', is_aligned=False)

        # 2. 计算 预测框 的能量矩阵 (DIoU Matrix) [cite: 135]
        # energy_pred = bbox_overlaps(pred_bboxes, pred_bboxes, mode='diou', is_aligned=False)


        energy_gt = calculate_diou_matrix(gt_bboxes, gt_bboxes)
        energy_pred = calculate_diou_matrix(pred_bboxes, pred_bboxes)
        
        # 3. 计算损失 (L_energy) [cite: 137, 140]
        # 我们只关心非对角线元素（物体之间的关系，而不是物体与自己）
        # 构建一个mask，去除对角线
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_bboxes.device)
        
        # 提取有效关系对
        valid_gt_energy = energy_gt[mask]
        valid_pred_energy = energy_pred[mask]

        # MSE Loss
        loss = nn.functional.mse_loss(valid_pred_energy, valid_gt_energy, reduction=self.reduction)

        return self.loss_weight * loss
    


    """
    
    def forward(self, pred_bboxes, gt_bboxes, gt_labels): # 注意多传一个 labels
        # ... (前文代码: 计算 energy_pred 和 energy_gt) ...
        
        # 1. 基础 Mask: 去掉对角线 (自己和自己比)
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        
        # 2. (可选) 进阶 Mask: 去掉同一物体的配对
        # gt_labels: [N] -> label_mat: [N, N]
        # label_mat[i, j] 为 True 表示 i 和 j 是同类/同物体
        label_mat = gt_labels.unsqueeze(0) == gt_labels.unsqueeze(1)
        
        # 我们只想保留 "不同物体" 之间的关系
        # diff_obj_mask = ~label_mat 
        
        # 最终 Mask (根据你的需求选择)
        # 选项 A: 保留所有 (你现在的逻辑) -> 包含了一致性约束
        final_mask = diag_mask 
        
        # 选项 B: 仅拓扑 (只算器官 A vs 器官 B) -> 纯粹的拓扑约束
        # final_mask = diag_mask & (~label_mat)

        loss = nn.functional.mse_loss(
            energy_pred[final_mask], 
            energy_gt[final_mask]
        )
    
    """