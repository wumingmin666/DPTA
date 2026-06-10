import numpy as np
import torch
from mmdet.structures import DetDataSample
from typing import List
from mmengine.structures import InstanceData
from pathlib import Path


def calculate_diou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate DIoU between two sets of bboxes.
    Args:
        bboxes1 (Tensor): bboxes of format (x1, y1, x2, y2), shape (n, 4).
        bboxes2 (Tensor): bboxes of format (x1, y1, x2, y2), shape (n, 4).
        eps (float): Eps to avoid division by zero.
    Returns:
        Tensor: DIoU values, shape (n,).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2.chunk(4, dim=-1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # print("=======================")
    # print(bboxes1.size())
    # print(bboxes2.size())

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw**2 + ch**2 + eps  # convex diagonal squared

    # Center distance
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    rho2 = (b2_cx - b1_cx)**2 + (b2_cy - b1_cy)**2  # center distance squared

    # DIoU取值范围[-1,1]
    dious = iou - rho2 / c2
    return dious


class GraphCorrectorLogPlus:
    """
    Improved GraphCorrector using normalized coordinates (0-1) and Log-space losses
    for Size and Aspect Ratio to handle scale differences better.
    """

    def __init__(self,
                 graph_path: str = "/media/Storage3/wmm/ICML/data/spineGE10/diou_graph_mean.npy",
                 size_stats_path: str = "/media/Storage3/wmm/ICML/data/spineGE10/size_stats.npz",
                 angle_graph_path: str = "/media/Storage3/wmm/ICML/data/spineGE10/angle_graph_mean.npy",
                 iterations: int = 80,
                 learning_rate: float = 0.0001,
                 loss_threshold: float = 0.01,
                 aspect_ratio_loss_weight: float = 1,
                 size_loss_weight: float = 0.2,
                 diou_loss_weight: float = 0.2,
                 angle_loss_weight: float = 0.2,
                 deviation_loss_weight: float = 0.4
                 ):

        # Load Mean and Variance stats
        # Assuming graph_path points to directory or we construct paths
        # For compatibility, let's assume the user passes the MEAN file path, and we infer the VAR file path
        # or we change the init to accept a directory.
        # Given the previous context, let's assume the user will update the paths or we handle it here.
        # To be safe and flexible, let's try to load variance if it exists, otherwise default to 1.0
        
        graph_path_obj = Path(graph_path)
        var_graph_path = graph_path_obj.parent / (graph_path_obj.stem.replace('_mean', '') + '_var.npy')
        if not var_graph_path.exists():
             # Try replacing 'graph' with 'graph_var' if naming convention differs
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

        self.iterations = iterations
        self.lr = learning_rate
        self.loss_threshold = loss_threshold
        self.aspect_ratio_loss_weight = aspect_ratio_loss_weight
        self.deviation_loss_weight = deviation_loss_weight
        self.size_loss_weight = size_loss_weight
        self.diou_loss_weight = diou_loss_weight
        self.angle_loss_weight = angle_loss_weight

        # Load Log-space stats
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

        self.count = 0

    def __call__(self, results_list: List[DetDataSample]) -> List[DetDataSample]:
        """
        Applies graph-based correction to a batch of detection results.
        """
        device = results_list[0].pred_instances['bboxes'].device
        self.avg_diou_graph = self.avg_diou_graph.to(device)
        self.var_diou_graph = self.var_diou_graph.to(device)
        self.avg_angle_cosine_graph = self.avg_angle_cosine_graph.to(device)
        self.var_angle_cosine_graph = self.var_angle_cosine_graph.to(device)
        self.avg_log_relative_size = self.avg_log_relative_size.to(device)
        self.var_log_relative_size = self.var_log_relative_size.to(device)
        self.avg_log_aspect_ratio = self.avg_log_aspect_ratio.to(device)
        self.var_log_aspect_ratio = self.var_log_aspect_ratio.to(device)

        for data_sample in results_list:
            pred_instances = data_sample.pred_instances
            if len(pred_instances.scores) == 0:
                continue

            # Step 1: Initial filtering by score threshold
            initial_keep_mask = pred_instances.scores > 0.001
            if not torch.any(initial_keep_mask):
                continue

            initial_keep_indices = torch.where(initial_keep_mask)[0]
            scores = pred_instances.scores[initial_keep_mask]
            labels = pred_instances.labels[initial_keep_mask]
            boxes = pred_instances.bboxes[initial_keep_mask]

            # Step 2: Select top-1 prediction for each class
            unique_labels = torch.unique(labels)
            final_keep_indices_in_filtered = []

            for label in unique_labels:
                class_mask = (labels == label)
                class_scores = scores[class_mask]
                top_score_idx_in_class = torch.argmax(class_scores)
                original_indices_for_class = torch.where(class_mask)[0]
                top_score_idx_in_filtered = original_indices_for_class[top_score_idx_in_class]
                final_keep_indices_in_filtered.append(top_score_idx_in_filtered)

            if not final_keep_indices_in_filtered:
                continue
            
            final_keep_indices_in_filtered = torch.tensor(final_keep_indices_in_filtered, device=labels.device, dtype=torch.long)

            pred_boxes = boxes[final_keep_indices_in_filtered]
            pred_scores = scores[final_keep_indices_in_filtered]
            pred_labels = labels[final_keep_indices_in_filtered]

            # Step 3: Boost low scores
            boost_mask = pred_scores < 0.3
            pred_scores[boost_mask] = 0.31

            # Refine boxes
            corrected_sample = {'scores': pred_scores, 'bboxes': pred_boxes, 'labels': pred_labels}
            with torch.enable_grad():
                corrected_sample = self._correct_single_image(
                    corrected_sample, data_sample.ori_shape)

            refined_boxes = corrected_sample['bboxes']
            
            # Update original instances
            if len(refined_boxes) == len(pred_boxes):
                final_absolute_indices = initial_keep_indices[final_keep_indices_in_filtered]
                original_bboxes = pred_instances.bboxes.clone()

                # Save pre-refine info
                pre_refine_bbox = original_bboxes[final_absolute_indices].clone()
                pre_refine_instances = InstanceData()
                pre_refine_instances.bboxes = pre_refine_bbox
                pre_refine_instances.scores = pred_scores
                pre_refine_instances.labels = pred_labels
                data_sample.pre_refine_instances = pre_refine_instances

                original_bboxes[final_absolute_indices] = refined_boxes
                pred_instances.bboxes = original_bboxes
                pred_instances.scores[final_absolute_indices] = pred_scores

        return results_list

    def _correct_single_image(self, pred_instances, img_wh):
        """Applies correction to the predictions of a single image using normalized coordinates."""
        bboxes = pred_instances['bboxes']
        labels = pred_instances['labels']
        scores = pred_instances['scores']
        
        img_h, img_w = img_wh
        
        # Normalize bboxes to 0-1 range: (cx, cy, w, h)
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2 / img_w
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2 / img_h
        w = (bboxes[:, 2] - bboxes[:, 0]) / img_w
        h = (bboxes[:, 3] - bboxes[:, 1]) / img_h
        
        # Clamp to avoid numerical issues with log
        w = torch.clamp(w, min=1e-4, max=1.0)
        h = torch.clamp(h, min=1e-4, max=1.0)
        
        xywh_norm = torch.stack([cx, cy, w, h], dim=-1)
        xywh_norm = xywh_norm.clone().detach().requires_grad_(True)
        initial_xywh_norm = xywh_norm.clone().detach()

        print("===========", self.count, "==========")
        self.count += 1

        for i in range(self.iterations):
            # 1. Convert normalized (cx, cy, w, h) back to absolute (x1, y1, x2, y2) for DIoU
            # We must use absolute coordinates (or aspect-ratio corrected) to match the priors
            # which were generated on absolute pixel coordinates. Normalized coordinates on non-square
            # images would distort the distance metric in DIoU.
            
            current_cx, current_cy, current_w, current_h = xywh_norm.unbind(-1)
            
            # Ensure w, h are positive for geometric validity
            current_w = torch.abs(current_w)
            current_h = torch.abs(current_h)
            
            # Convert to absolute coordinates for DIoU calculation
            abs_cx = current_cx * img_w
            abs_cy = current_cy * img_h
            
            # Decoupled Optimization:
            # Detach w and h for Pair Loss calculation.
            # This forces Pair Loss to only optimize position (cx, cy),
            # preventing it from distorting the shape to satisfy structural constraints.
            # abs_w = (current_w * img_w).detach()
            # abs_h = (current_h * img_h).detach()
            abs_w = current_w * img_w
            abs_h = current_h * img_h
            
            
            x1 = abs_cx - abs_w / 2
            y1 = abs_cy - abs_h / 2
            x2 = abs_cx + abs_w / 2
            y2 = abs_cy + abs_h / 2
            
            current_bboxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

            n_boxes = len(current_bboxes_abs)
            if n_boxes < 2:
                break
                
            idx_pairs = torch.combinations(torch.arange(n_boxes, device=xywh_norm.device), r=2)
            
            bboxes1 = current_bboxes_abs[idx_pairs[:, 0]]
            bboxes2 = current_bboxes_abs[idx_pairs[:, 1]]
            
            labels1 = labels[idx_pairs[:, 0]]
            labels2 = labels[idx_pairs[:, 1]]
            
            scores1 = scores[idx_pairs[:, 0]]
            scores2 = scores[idx_pairs[:, 1]]

            # --- Pairwise DIoU Loss ---
            # We use the existing calculate_diou. It works with any scale.
            current_dious = calculate_diou(bboxes1, bboxes2).squeeze()
            expected_dious = self.avg_diou_graph[labels1, labels2]
            var_dious = self.var_diou_graph[labels1, labels2]
            
            # Inverse Variance Weighting: (x - mu)^2 / (2 * sigma^2)
            # We add a small epsilon to variance to avoid division by zero
            # We also keep the confidence weighting
            # Removed *100 scaling factor to balance with Log-Space Size/AR losses
            pair_losses = ((1-scores1) * (1-scores2) * (current_dious - expected_dious)**2) / (2 * var_dious + 1e-6)
            
            # Normalize by (n_boxes - 1) to ensure the gradient magnitude per box
            # represents the "average" force from neighbors, rather than growing linearly with N.
            if n_boxes > 1:
                pair_losses_sum = pair_losses.sum() / (n_boxes - 1)
            else:
                pair_losses_sum = pair_losses.sum()

            low_score_mask = scores < 0.5

            # --- Log-Space Size Loss ---
            # current_w, current_h are normalized relative to image size
            # relative_size = w * h (since normalized)
            current_rel_size = current_w * current_h
            log_current_size = torch.log(current_rel_size + 1e-7)
            expected_log_size = self.avg_log_relative_size.to(labels.device)[labels]
            var_log_size = self.var_log_relative_size.to(labels.device)[labels]
            
            size_loss = (((log_current_size[low_score_mask] - expected_log_size[low_score_mask])**2) / (2 * var_log_size[low_score_mask] + 1e-6)).sum()

            # --- Log-Space Aspect Ratio Loss ---
            # AR = w_abs / h_abs = (w_norm * img_w) / (h_norm * img_h)
            # log(AR) = log(w_norm/h_norm) + log(img_w/img_h)
            # But our stats are likely collected as w/h.
            # If we use normalized w, h, we must adjust for image aspect ratio if the prior was collected on absolute pixels.
            # The prior generation script calculates AR = w_abs / h_abs.
            # So here: current_ar = (current_w * img_w) / (current_h * img_h)
            
            current_ar = (current_w * img_w) / (current_h * img_h + 1e-7)
            log_current_ar = torch.log(current_ar + 1e-7)
            expected_log_ar = self.avg_log_aspect_ratio.to(labels.device)[labels]
            var_log_ar = self.var_log_aspect_ratio.to(labels.device)[labels]
            
            aspect_ratio_loss = (((log_current_ar[low_score_mask] - expected_log_ar[low_score_mask])**2) / (2 * var_log_ar[low_score_mask] + 1e-6)).sum()

            # --- Deviation Loss ---
            # Penalize moving too far from initial prediction
            # Re-added *100 scaling factor to match magnitude of Log-Space losses
            deviation_loss = ((xywh_norm - initial_xywh_norm)**2).sum() * 100

            # --- Angle Loss ---
            # Need to pass absolute or consistent coordinates. Normalized (cx, cy) is fine if aspect ratio is handled?
            # Angle depends on geometry. If image is non-square, normalized space distorts angles.
            # We should convert to absolute pixel coordinates for angle calculation to match physical reality/prior.
            
            abs_cx = current_cx * img_w
            abs_cy = current_cy * img_h
            abs_xywh = torch.stack([abs_cx, abs_cy, current_w * img_w, current_h * img_h], dim=-1)
            
            angle_loss = self._calculate_triangle_loss(abs_xywh, scores, labels)

            # --- Total Energy ---
            energy = (self.diou_loss_weight * pair_losses_sum +
                      self.size_loss_weight * size_loss +
                      self.aspect_ratio_loss_weight * aspect_ratio_loss +
                      self.deviation_loss_weight * deviation_loss +
                      self.angle_loss_weight * angle_loss)

            if energy < 1e-3:
                break

            energy.backward()

            # if i < 50:
            if i == self.iterations - 1 or i == 0:
                print(f"Iter {i}, Energy: {energy.item():.4f}, Size: {size_loss.item():.4f}, AR: {aspect_ratio_loss.item():.4f}, Pair: {pair_losses_sum.item():.4f}, Dev: {deviation_loss.item():.4f}, Angle: {angle_loss.item():.4f}")

            # Update
            with torch.no_grad():
                grad = xywh_norm.grad
                if grad is not None:
                    # Dynamic learning rate
                    learning_rates = torch.zeros_like(scores)
                    lr_update_mask = scores < 0.6
                    if lr_update_mask.any():
                        selected_scores = scores[lr_update_mask]
                        scaled_lr = self.lr * (1 - selected_scores)
                        learning_rates[lr_update_mask] = scaled_lr
                    
                    # Apply gradient scaling for w and h to prevent oscillation
                    # cx, cy: scale 1.0
                    # w, h: scale 0.01 (dampening factor)
                    grad_scale = torch.tensor([1.0, 1.0, 0.01, 0.01], device=grad.device)
                    
                    update_step = learning_rates.unsqueeze(1) * grad * grad_scale
                    
                    # Gradient Clipping: Limit update step to 5% of image size to prevent explosion
                    # caused by small variances in Inverse Variance Weighting
                    update_step = torch.clamp(update_step, min=-0.05, max=0.05)
                    
                    xywh_norm -= update_step
                    grad.zero_()
                    
                    # Clamp to valid range (0-1)
                    xywh_norm[:, 0].clamp_(0, 1) # cx
                    xywh_norm[:, 1].clamp_(0, 1) # cy
                    xywh_norm[:, 2].clamp_(1e-4, 1) # w
                    xywh_norm[:, 3].clamp_(1e-4, 1) # h

        # Convert final normalized (cx, cy, w, h) back to absolute (x1, y1, x2, y2)
        final_cx, final_cy, final_w, final_h = xywh_norm.detach().unbind(-1)
        
        final_x1 = (final_cx - final_w / 2) * img_w
        final_y1 = (final_cy - final_h / 2) * img_h
        final_x2 = (final_cx + final_w / 2) * img_w
        final_y2 = (final_cy + final_h / 2) * img_h
        
        corrected_bboxes = torch.stack([final_x1, final_y1, final_x2, final_y2], dim=-1)
        
        pred_instances['bboxes'] = corrected_bboxes
        return pred_instances

    def _calculate_triangle_loss(self, xywh, scores, labels):
        """
        Calculates the loss based on the shape (angles) of the triangle formed by
        the top-3 highest confidence predictions.
        xywh: Absolute coordinates (cx, cy, w, h)
        """
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
        
        # Angle at P0
        current_cos_0 = get_cosine_similarity(p0, p1, p2)
        expected_cos_0 = self.avg_angle_cosine_graph[l0, l1, l2]
        var_cos_0 = self.var_angle_cosine_graph[l0, l1, l2]
        # Clamp variance to avoid exploding loss due to sparse data (min var = 0.05)
        var_cos_0 = torch.clamp(var_cos_0, min=0.05)
        loss += (current_cos_0 - expected_cos_0) ** 2 / (2 * var_cos_0)

        # Angle at P1
        current_cos_1 = get_cosine_similarity(p1, p0, p2)
        expected_cos_1 = self.avg_angle_cosine_graph[l1, l0, l2]
        var_cos_1 = self.var_angle_cosine_graph[l1, l0, l2]
        var_cos_1 = torch.clamp(var_cos_1, min=0.05)
        loss += (current_cos_1 - expected_cos_1) ** 2 / (2 * var_cos_1)

        # Angle at P2
        current_cos_2 = get_cosine_similarity(p2, p0, p1)
        expected_cos_2 = self.avg_angle_cosine_graph[l2, l0, l1]
        var_cos_2 = self.var_angle_cosine_graph[l2, l0, l1]
        var_cos_2 = torch.clamp(var_cos_2, min=0.05)
        loss += (current_cos_2 - expected_cos_2) ** 2 / (2 * var_cos_2)

        return loss


class GraphCorrectorLogPlus:
    """
    Improved GraphCorrector using normalized coordinates (0-1) and Log-space losses
    for Size and Aspect Ratio to handle scale differences better.
    """

    def __init__(self,
                 graph_path: str = "/media/Storage3/wmm/ICML/data/spineGE10/diou_graph_mean.npy",
                 size_stats_path: str = "/media/Storage3/wmm/ICML/data/spineGE10/size_stats.npz",
                 angle_graph_path: str = "/media/Storage3/wmm/ICML/data/spineGE10/angle_graph_mean.npy",
                 iterations: int = 80,
                 learning_rate: float = 0.0001,
                 loss_threshold: float = 0.01,
                 aspect_ratio_loss_weight: float = 1,
                 size_loss_weight: float = 0.2,
                 diou_loss_weight: float = 0.2,
                 angle_loss_weight: float = 0.2,
                 deviation_loss_weight: float = 0.4
                 ):

        # Load Mean and Variance stats
        # Assuming graph_path points to directory or we construct paths
        # For compatibility, let's assume the user passes the MEAN file path, and we infer the VAR file path
        # or we change the init to accept a directory.
        # Given the previous context, let's assume the user will update the paths or we handle it here.
        # To be safe and flexible, let's try to load variance if it exists, otherwise default to 1.0
        
        graph_path_obj = Path(graph_path)
        var_graph_path = graph_path_obj.parent / (graph_path_obj.stem.replace('_mean', '') + '_var.npy')
        if not var_graph_path.exists():
             # Try replacing 'graph' with 'graph_var' if naming convention differs
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

        self.iterations = iterations
        self.lr = learning_rate
        self.loss_threshold = loss_threshold
        self.aspect_ratio_loss_weight = aspect_ratio_loss_weight
        self.deviation_loss_weight = deviation_loss_weight
        self.size_loss_weight = size_loss_weight
        self.diou_loss_weight = diou_loss_weight
        self.angle_loss_weight = angle_loss_weight

        # Load Log-space stats
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

        self.count = 0

    def __call__(self, results_list: List[DetDataSample]) -> List[DetDataSample]:
        """
        Applies graph-based correction to a batch of detection results.
        """
        device = results_list[0].pred_instances['bboxes'].device
        self.avg_diou_graph = self.avg_diou_graph.to(device)
        self.var_diou_graph = self.var_diou_graph.to(device)
        self.avg_angle_cosine_graph = self.avg_angle_cosine_graph.to(device)
        self.var_angle_cosine_graph = self.var_angle_cosine_graph.to(device)
        self.avg_log_relative_size = self.avg_log_relative_size.to(device)
        self.var_log_relative_size = self.var_log_relative_size.to(device)
        self.avg_log_aspect_ratio = self.avg_log_aspect_ratio.to(device)
        self.var_log_aspect_ratio = self.var_log_aspect_ratio.to(device)

        for data_sample in results_list:
            pred_instances = data_sample.pred_instances
            if len(pred_instances.scores) == 0:
                continue

            # Step 1: Initial filtering by score threshold
            initial_keep_mask = pred_instances.scores > 0.001
            if not torch.any(initial_keep_mask):
                continue

            initial_keep_indices = torch.where(initial_keep_mask)[0]
            scores = pred_instances.scores[initial_keep_mask]
            labels = pred_instances.labels[initial_keep_mask]
            boxes = pred_instances.bboxes[initial_keep_mask]

            # Step 2: Select top-1 prediction for each class
            unique_labels = torch.unique(labels)
            final_keep_indices_in_filtered = []

            for label in unique_labels:
                class_mask = (labels == label)
                class_scores = scores[class_mask]
                top_score_idx_in_class = torch.argmax(class_scores)
                original_indices_for_class = torch.where(class_mask)[0]
                top_score_idx_in_filtered = original_indices_for_class[top_score_idx_in_class]
                final_keep_indices_in_filtered.append(top_score_idx_in_filtered)

            if not final_keep_indices_in_filtered:
                continue
            
            final_keep_indices_in_filtered = torch.tensor(final_keep_indices_in_filtered, device=labels.device, dtype=torch.long)

            pred_boxes = boxes[final_keep_indices_in_filtered]
            pred_scores = scores[final_keep_indices_in_filtered]
            pred_labels = labels[final_keep_indices_in_filtered]

            # Step 3: Boost low scores
            boost_mask = pred_scores < 0.3
            pred_scores[boost_mask] = 0.31

            # Refine boxes
            corrected_sample = {'scores': pred_scores, 'bboxes': pred_boxes, 'labels': pred_labels}
            with torch.enable_grad():
                corrected_sample = self._correct_single_image(
                    corrected_sample, data_sample.ori_shape)

            refined_boxes = corrected_sample['bboxes']
            
            # Update original instances
            if len(refined_boxes) == len(pred_boxes):
                final_absolute_indices = initial_keep_indices[final_keep_indices_in_filtered]
                original_bboxes = pred_instances.bboxes.clone()

                # Save pre-refine info
                pre_refine_bbox = original_bboxes[final_absolute_indices].clone()
                pre_refine_instances = InstanceData()
                pre_refine_instances.bboxes = pre_refine_bbox
                pre_refine_instances.scores = pred_scores
                pre_refine_instances.labels = pred_labels
                data_sample.pre_refine_instances = pre_refine_instances

                original_bboxes[final_absolute_indices] = refined_boxes
                pred_instances.bboxes = original_bboxes
                pred_instances.scores[final_absolute_indices] = pred_scores

        return results_list

    def _correct_single_image(self, pred_instances, img_wh):
        """Applies correction to the predictions of a single image using normalized coordinates."""
        bboxes = pred_instances['bboxes']
        labels = pred_instances['labels']
        scores = pred_instances['scores']
        
        img_h, img_w = img_wh
        
        # Normalize bboxes to 0-1 range: (cx, cy, w, h)
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2 / img_w
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2 / img_h
        w = (bboxes[:, 2] - bboxes[:, 0]) / img_w
        h = (bboxes[:, 3] - bboxes[:, 1]) / img_h
        
        # Clamp to avoid numerical issues with log
        w = torch.clamp(w, min=1e-4, max=1.0)
        h = torch.clamp(h, min=1e-4, max=1.0)
        
        xywh_norm = torch.stack([cx, cy, w, h], dim=-1)
        xywh_norm = xywh_norm.clone().detach().requires_grad_(True)
        initial_xywh_norm = xywh_norm.clone().detach()

        print("===========", self.count, "==========")
        self.count += 1

        for i in range(self.iterations):
            # 1. Convert normalized (cx, cy, w, h) back to absolute (x1, y1, x2, y2) for DIoU
            # We must use absolute coordinates (or aspect-ratio corrected) to match the priors
            # which were generated on absolute pixel coordinates. Normalized coordinates on non-square
            # images would distort the distance metric in DIoU.
            
            current_cx, current_cy, current_w, current_h = xywh_norm.unbind(-1)
            
            # Ensure w, h are positive for geometric validity
            current_w = torch.abs(current_w)
            current_h = torch.abs(current_h)
            
            # Convert to absolute coordinates for DIoU calculation
            abs_cx = current_cx * img_w
            abs_cy = current_cy * img_h
            
            # Decoupled Optimization:
            # Detach w and h for Pair Loss calculation.
            # This forces Pair Loss to only optimize position (cx, cy),
            # preventing it from distorting the shape to satisfy structural constraints.
            # abs_w = (current_w * img_w).detach()
            # abs_h = (current_h * img_h).detach()
            abs_w = current_w * img_w
            abs_h = current_h * img_h
            
            
            x1 = abs_cx - abs_w / 2
            y1 = abs_cy - abs_h / 2
            x2 = abs_cx + abs_w / 2
            y2 = abs_cy + abs_h / 2
            
            current_bboxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

            n_boxes = len(current_bboxes_abs)
            if n_boxes < 2:
                break
                
            idx_pairs = torch.combinations(torch.arange(n_boxes, device=xywh_norm.device), r=2)
            
            bboxes1 = current_bboxes_abs[idx_pairs[:, 0]]
            bboxes2 = current_bboxes_abs[idx_pairs[:, 1]]
            
            labels1 = labels[idx_pairs[:, 0]]
            labels2 = labels[idx_pairs[:, 1]]
            
            scores1 = scores[idx_pairs[:, 0]]
            scores2 = scores[idx_pairs[:, 1]]

            # --- Pairwise DIoU Loss ---
            # We use the existing calculate_diou. It works with any scale.
            current_dious = calculate_diou(bboxes1, bboxes2).squeeze()
            expected_dious = self.avg_diou_graph[labels1, labels2]
            var_dious = self.var_diou_graph[labels1, labels2]
            
            # Inverse Variance Weighting: (x - mu)^2 / (2 * sigma^2)
            # We add a small epsilon to variance to avoid division by zero
            # We also keep the confidence weighting
            # Removed *100 scaling factor to balance with Log-Space Size/AR losses
            pair_losses = ((1-scores1) * (1-scores2) * (current_dious - expected_dious)**2) / (2 * var_dious + 1e-6)
            
            # Normalize by (n_boxes - 1) to ensure the gradient magnitude per box
            # represents the "average" force from neighbors, rather than growing linearly with N.
            if n_boxes > 1:
                pair_losses_sum = pair_losses.sum() / (n_boxes - 1)
            else:
                pair_losses_sum = pair_losses.sum()

            low_score_mask = scores < 0.5

            # --- Log-Space Size Loss ---
            # current_w, current_h are normalized relative to image size
            # relative_size = w * h (since normalized)
            current_rel_size = current_w * current_h
            log_current_size = torch.log(current_rel_size + 1e-7)
            expected_log_size = self.avg_log_relative_size.to(labels.device)[labels]
            var_log_size = self.var_log_relative_size.to(labels.device)[labels]
            
            size_loss = (((log_current_size[low_score_mask] - expected_log_size[low_score_mask])**2) / (2 * var_log_size[low_score_mask] + 1e-6)).sum()

            # --- Log-Space Aspect Ratio Loss ---
            # AR = w_abs / h_abs = (w_norm * img_w) / (h_norm * img_h)
            # log(AR) = log(w_norm/h_norm) + log(img_w/img_h)
            # But our stats are likely collected as w/h.
            # If we use normalized w, h, we must adjust for image aspect ratio if the prior was collected on absolute pixels.
            # The prior generation script calculates AR = w_abs / h_abs.
            # So here: current_ar = (current_w * img_w) / (current_h * img_h)
            
            current_ar = (current_w * img_w) / (current_h * img_h + 1e-7)
            log_current_ar = torch.log(current_ar + 1e-7)
            expected_log_ar = self.avg_log_aspect_ratio.to(labels.device)[labels]
            var_log_ar = self.var_log_aspect_ratio.to(labels.device)[labels]
            
            aspect_ratio_loss = (((log_current_ar[low_score_mask] - expected_log_ar[low_score_mask])**2) / (2 * var_log_ar[low_score_mask] + 1e-6)).sum()


            # --- Angle Loss ---
            # Need to pass absolute or consistent coordinates. Normalized (cx, cy) is fine if aspect ratio is handled?
            # Angle depends on geometry. If image is non-square, normalized space distorts angles.
            # We should convert to absolute pixel coordinates for angle calculation to match physical reality/prior.
            
            abs_cx = current_cx * img_w
            abs_cy = current_cy * img_h
            abs_xywh = torch.stack([abs_cx, abs_cy, current_w * img_w, current_h * img_h], dim=-1)
            
            angle_loss = self._calculate_triangle_loss(abs_xywh, scores, labels)

            # --- Total Energy ---
            energy = (self.diou_loss_weight * pair_losses_sum +
                      self.size_loss_weight * size_loss +
                      self.aspect_ratio_loss_weight * aspect_ratio_loss +
                      self.deviation_loss_weight * deviation_loss +
                      self.angle_loss_weight * angle_loss)

            if energy < 1e-3:
                break

            energy.backward()

            # if i < 50:
            if i == self.iterations - 1 or i == 0:
                print(f"Iter {i}, Energy: {energy.item():.4f}, Size: {size_loss.item():.4f}, AR: {aspect_ratio_loss.item():.4f}, Pair: {pair_losses_sum.item():.4f}, Dev: {deviation_loss.item():.4f}, Angle: {angle_loss.item():.4f}")

            # Update
            with torch.no_grad():
                grad = xywh_norm.grad
                if grad is not None:
                    # Dynamic learning rate
                    learning_rates = torch.zeros_like(scores)
                    lr_update_mask = scores < 0.6
                    if lr_update_mask.any():
                        selected_scores = scores[lr_update_mask]
                        scaled_lr = self.lr * (1 - selected_scores)
                        learning_rates[lr_update_mask] = scaled_lr
                    
                    # Apply gradient scaling for w and h to prevent oscillation
                    # cx, cy: scale 1.0
                    # w, h: scale 0.01 (dampening factor)
                    grad_scale = torch.tensor([1.0, 1.0, 0.01, 0.01], device=grad.device)
                    
                    update_step = learning_rates.unsqueeze(1) * grad * grad_scale
                    
                    # Gradient Clipping: Limit update step to 5% of image size to prevent explosion
                    # caused by small variances in Inverse Variance Weighting
                    update_step = torch.clamp(update_step, min=-0.05, max=0.05)
                    
                    xywh_norm -= update_step
                    grad.zero_()
                    
                    # Clamp to valid range (0-1)
                    xywh_norm[:, 0].clamp_(0, 1) # cx
                    xywh_norm[:, 1].clamp_(0, 1) # cy
                    xywh_norm[:, 2].clamp_(1e-4, 1) # w
                    xywh_norm[:, 3].clamp_(1e-4, 1) # h

        # Convert final normalized (cx, cy, w, h) back to absolute (x1, y1, x2, y2)
        final_cx, final_cy, final_w, final_h = xywh_norm.detach().unbind(-1)
        
        final_x1 = (final_cx - final_w / 2) * img_w
        final_y1 = (final_cy - final_h / 2) * img_h
        final_x2 = (final_cx + final_w / 2) * img_w
        final_y2 = (final_cy + final_h / 2) * img_h
        
        corrected_bboxes = torch.stack([final_x1, final_y1, final_x2, final_y2], dim=-1)
        
        pred_instances['bboxes'] = corrected_bboxes
        return pred_instances

    def _calculate_triangle_loss(self, xywh, scores, labels):
        """
        Calculates the loss based on the shape (angles) of the triangle formed by
        the top-3 highest confidence predictions.
        xywh: Absolute coordinates (cx, cy, w, h)
        """
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
        
        # Angle at P0
        current_cos_0 = get_cosine_similarity(p0, p1, p2)
        expected_cos_0 = self.avg_angle_cosine_graph[l0, l1, l2]
        var_cos_0 = self.var_angle_cosine_graph[l0, l1, l2]
        # Clamp variance to avoid exploding loss due to sparse data (min var = 0.05)
        var_cos_0 = torch.clamp(var_cos_0, min=0.05)
        loss += (current_cos_0 - expected_cos_0) ** 2 / (2 * var_cos_0)

        # Angle at P1
        current_cos_1 = get_cosine_similarity(p1, p0, p2)
        expected_cos_1 = self.avg_angle_cosine_graph[l1, l0, l2]
        var_cos_1 = self.var_angle_cosine_graph[l1, l0, l2]
        var_cos_1 = torch.clamp(var_cos_1, min=0.05)
        loss += (current_cos_1 - expected_cos_1) ** 2 / (2 * var_cos_1)

        # Angle at P2
        current_cos_2 = get_cosine_similarity(p2, p0, p1)
        expected_cos_2 = self.avg_angle_cosine_graph[l2, l0, l1]
        var_cos_2 = self.var_angle_cosine_graph[l2, l0, l1]
        var_cos_2 = torch.clamp(var_cos_2, min=0.05)
        loss += (current_cos_2 - expected_cos_2) ** 2 / (2 * var_cos_2)

        return loss
