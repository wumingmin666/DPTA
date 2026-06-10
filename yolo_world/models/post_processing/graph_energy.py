import numpy as np
import torch
from typing import List
from pathlib import Path
from mmengine.structures import InstanceData

# 本文件提供训练阶段使用的能量计算器（GraphEnergyLogPlus）。
# 说明：
# - 该计算器只负责计算能量（energy）损失，并返回一个标量损失值，
#   不会修改预测框（bbox）。
# - 需要传入由 head.predict_by_feat / predict 返回的 InstanceData 列表
#  （每项包含 bboxes, scores, labels），以及与之对应的 batch_img_metas，
#   以便将 bbox 恢复到原始图像坐标用于尺寸/长宽比统计。
# - 本实现对输入做了容错处理：会尽量从预测本身推断设备并把统计量移到相同设备。


def calculate_diou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2.chunk(4, dim=-1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + eps
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    rho2 = (b2_cx - b1_cx)**2 + (b2_cy - b1_cy)**2
    dious = iou - rho2 / c2
    return dious


class GraphEnergyLogPlus:
    """
    训练阶段使用的能量计算器（不修改 bbox），输出 batch 上的能量损失。
    """

    def __init__(self,
                 graph_path: str,
                 size_stats_path: str,
                 angle_graph_path: str,
                 aspect_ratio_loss_weight: float = 0.3,
                 size_loss_weight: float = 0.2,
                 diou_loss_weight: float = 0.5,
                 angle_loss_weight: float = 0.0,
                 deviation_loss_weight: float = 0.0):
        graph_path_obj = Path(graph_path)
        var_graph_path = graph_path_obj.parent / (graph_path_obj.stem.replace('_mean', '') + '_var.npy')
        if not var_graph_path.exists():
            var_graph_path = graph_path_obj.parent / 'diou_graph_var.npy'

        self.avg_diou_graph = torch.from_numpy(np.load(graph_path)).float()
        if var_graph_path.exists():
            self.var_diou_graph = torch.from_numpy(np.load(var_graph_path)).float()
        else:
            self.var_diou_graph = torch.ones_like(self.avg_diou_graph)

        angle_graph_path_obj = Path(angle_graph_path)
        var_angle_path = angle_graph_path_obj.parent / (angle_graph_path_obj.stem.replace('_mean', '') + '_var.npy')
        if not var_angle_path.exists():
            var_angle_path = angle_graph_path_obj.parent / 'angle_graph_var.npy'

        self.avg_angle_cosine_graph = torch.from_numpy(np.load(angle_graph_path)).float()
        if var_angle_path.exists():
            self.var_angle_cosine_graph = torch.from_numpy(np.load(var_angle_path)).float()
        else:
            self.var_angle_cosine_graph = torch.ones_like(self.avg_angle_cosine_graph)

        size_stats = np.load(size_stats_path)
        self.avg_log_relative_size = torch.from_numpy(size_stats['avg_log_relative_size']).float()
        self.avg_log_aspect_ratio = torch.from_numpy(size_stats['avg_log_aspect_ratio']).float()
        if 'var_log_relative_size' in size_stats:
            self.var_log_relative_size = torch.from_numpy(size_stats['var_log_relative_size']).float()
        else:
            self.var_log_relative_size = torch.ones_like(self.avg_log_relative_size)
        if 'var_log_aspect_ratio' in size_stats:
            self.var_log_aspect_ratio = torch.from_numpy(size_stats['var_log_aspect_ratio']).float()
        else:
            self.var_log_aspect_ratio = torch.ones_like(self.avg_log_aspect_ratio)

        self.aspect_ratio_loss_weight = aspect_ratio_loss_weight
        self.size_loss_weight = size_loss_weight
        self.diou_loss_weight = diou_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.deviation_loss_weight = deviation_loss_weight

    def compute_batch_energy(self, results_list: List[InstanceData], batch_img_metas: List[dict]) -> torch.Tensor:
        """Compute total energy loss for a batch of predictions.

        Args:
            results_list: list of InstanceData (each with bboxes, scores, labels)
            batch_img_metas: list of image meta dicts (must contain 'ori_shape')
        Returns:
            scalar tensor (sum energy over batch)
        """
        # 注意：batch_img_metas 中每项应至少包含 'ori_shape' (H,W,...)，
        # 用于将预测框的归一化坐标恢复为像素级坐标并计算尺寸/长宽比等。
        # 如果缺失，将会导致退化估计（见 caller 端需尽量保证传入准确 metainfo）。
        if len(results_list) == 0:
            return torch.tensor(0., requires_grad=True)

        device = None
        # move priors to device lazily
        for ds in results_list:
            if hasattr(ds, 'bboxes') and ds.bboxes is not None:
                device = ds.bboxes.device
                break
        if device is None:
            device = torch.device('cpu')

        self.avg_diou_graph = self.avg_diou_graph.to(device)
        self.var_diou_graph = self.var_diou_graph.to(device)
        self.avg_angle_cosine_graph = self.avg_angle_cosine_graph.to(device)
        self.var_angle_cosine_graph = self.var_angle_cosine_graph.to(device)
        self.avg_log_relative_size = self.avg_log_relative_size.to(device)
        self.var_log_relative_size = self.var_log_relative_size.to(device)
        self.avg_log_aspect_ratio = self.avg_log_aspect_ratio.to(device)
        self.var_log_aspect_ratio = self.var_log_aspect_ratio.to(device)

        total_energy = torch.tensor(0., device=device)
        for pred, meta in zip(results_list, batch_img_metas):
            if not hasattr(pred, 'scores') or pred.scores.numel() == 0:
                continue

            # 取出预测分数、类别、bbox，并从 meta 中获取原始图像尺寸
            scores = pred.scores
            labels = pred.labels
            boxes = pred.bboxes
            # 期望 meta['ori_shape'] 为 (H, W, C) 或 (H, W)
            img_h, img_w = meta['ori_shape'][:2]

            # 将 bbox 转为归一化的 (cx, cy, w, h)，范围大致 0-1
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h
            w = (boxes[:, 2] - boxes[:, 0]) / img_w
            h = (boxes[:, 3] - boxes[:, 1]) / img_h
            w = torch.clamp(w, min=1e-7)
            h = torch.clamp(h, min=1e-7)

            n_boxes = len(boxes)
            if n_boxes < 2:
                continue

            idx_pairs = torch.combinations(torch.arange(n_boxes, device=device), r=2)
            bboxes1 = boxes[idx_pairs[:, 0]]
            bboxes2 = boxes[idx_pairs[:, 1]]
            labels1 = labels[idx_pairs[:, 0]]
            labels2 = labels[idx_pairs[:, 1]]
            scores1 = scores[idx_pairs[:, 0]]
            scores2 = scores[idx_pairs[:, 1]]

            # 计算成对 DIoU 并与先验期望比较得到 pairwise 损失
            current_dious = calculate_diou(bboxes1, bboxes2).squeeze()
            expected_dious = self.avg_diou_graph[labels1, labels2]
            var_dious = self.var_diou_graph[labels1, labels2]
            pair_losses = ((1-scores1) * (1-scores2) * (current_dious - expected_dious)**2) / (2 * var_dious + 1e-6)
            if n_boxes > 1:
                pair_losses_sum = pair_losses.sum() / (n_boxes - 1)
            else:
                pair_losses_sum = pair_losses.sum()

            low_score_mask = scores < 0.9

            current_rel_size = w * h
            log_current_size = torch.log(current_rel_size + 1e-7)
            expected_log_size = self.avg_log_relative_size.to(labels.device)[labels]
            var_log_size = self.var_log_relative_size.to(labels.device)[labels]
            size_loss = (((log_current_size[low_score_mask] - expected_log_size[low_score_mask])**2) / (2 * var_log_size[low_score_mask] + 1e-6)).sum()

            current_ar = (w * img_w) / (h * img_h + 1e-7)
            log_current_ar = torch.log(current_ar + 1e-7)
            expected_log_ar = self.avg_log_aspect_ratio.to(labels.device)[labels]
            var_log_ar = self.var_log_aspect_ratio.to(labels.device)[labels]
            aspect_ratio_loss = (((log_current_ar[low_score_mask] - expected_log_ar[low_score_mask])**2) / (2 * var_log_ar[low_score_mask] + 1e-6)).sum()

            # 基于置信度最高的 top-3 点计算角度损失（这里使用像素坐标）
            angle_loss = self._calculate_triangle_loss(torch.stack([cx*img_w, cy*img_h, w*img_w, h*img_h], dim=-1), scores, labels)

            # 综合各项损失得到最终能量（energy）标量
            energy = (self.diou_loss_weight * pair_losses_sum +
                      self.size_loss_weight * size_loss +
                      self.aspect_ratio_loss_weight * aspect_ratio_loss +
                      self.angle_loss_weight * angle_loss)
            print(f"Energy components - DIoU: {pair_losses_sum.item():.4f}, Size: {size_loss.item():.4f}, Aspect Ratio: {aspect_ratio_loss.item():.4f}, Angle: {angle_loss.item():.4f}")
            total_energy = total_energy + energy

        return total_energy

    def _calculate_triangle_loss(self, xywh, scores, labels):
        if len(scores) < 3:
            return torch.tensor(0.0, device=xywh.device)
        _, top_indices = torch.topk(scores, 3)
        top_centers = xywh[top_indices, :2]
        top_labels = labels[top_indices]
        p0, p1, p2 = top_centers[0], top_centers[1], top_centers[2]
        l0, l1, l2 = top_labels[0], top_labels[1], top_labels[2]

        def get_cosine_similarity(center, p_a, p_b):
            v1 = p_a - center
            v2 = p_b - center
            v1 = v1 / (v1.norm() + 1e-7)
            v2 = v2 / (v2.norm() + 1e-7)
            return torch.dot(v1, v2)

        loss = torch.tensor(0.0, device=xywh.device)
        current_cos_0 = get_cosine_similarity(p0, p1, p2)
        expected_cos_0 = self.avg_angle_cosine_graph[l0, l1, l2]
        var_cos_0 = self.var_angle_cosine_graph[l0, l1, l2]
        var_cos_0 = torch.clamp(var_cos_0, min=0.05)
        loss = loss + (current_cos_0 - expected_cos_0) ** 2 / (2 * var_cos_0)

        current_cos_1 = get_cosine_similarity(p1, p0, p2)
        expected_cos_1 = self.avg_angle_cosine_graph[l1, l0, l2]
        var_cos_1 = self.var_angle_cosine_graph[l1, l0, l2]
        var_cos_1 = torch.clamp(var_cos_1, min=0.05)
        loss = loss + (current_cos_1 - expected_cos_1) ** 2 / (2 * var_cos_1)

        current_cos_2 = get_cosine_similarity(p2, p0, p1)
        expected_cos_2 = self.avg_angle_cosine_graph[l2, l0, l1]
        var_cos_2 = self.var_angle_cosine_graph[l2, l0, l1]
        var_cos_2 = torch.clamp(var_cos_2, min=0.05)
        loss = loss + (current_cos_2 - expected_cos_2) ** 2 / (2 * var_cos_2)

        return loss
