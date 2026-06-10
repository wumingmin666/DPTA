# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.dataset import COLLATE_FUNCTIONS


@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    batch_img_metas = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        # gt_bboxes = datasamples.gt_instances.bboxes
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)
        # 收集每张样本的 metainfo（用于后续恢复到原始图像坐标等）
        if hasattr(datasamples, 'metainfo'):
            batch_img_metas.append(datasamples.metainfo)
        else:
            # 若没有 metainfo，则尽量从输入推断最小信息以保证兼容性
            if hasattr(inputs, 'shape'):
                # inputs 可能是 Tensor 或其它容器，常见为 CHW 或 list
                try:
                    c, h, w = inputs.shape[-3:]
                    batch_img_metas.append({'ori_shape': (h, w, c), 'scale_factor': 1.0})
                except Exception:
                    batch_img_metas.append({})
            else:
                batch_img_metas.append({})

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    # 将 img_metas 一并返回，便于后续在 model 中恢复到原始图像坐标
    collated_results['data_samples']['img_metas'] = batch_img_metas
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)

    if hasattr(data_batch[0]['data_samples'], 'texts'):
        batch_texts = [meta['data_samples'].texts for meta in data_batch]
        collated_results['data_samples']['texts'] = batch_texts

    if hasattr(data_batch[0]['data_samples'], 'is_detection'):
        # detection flag
        batch_detection = [meta['data_samples'].is_detection
                           for meta in data_batch]
        collated_results['data_samples']['is_detection'] = torch.tensor(
            batch_detection)
    # print("Collate fn output keys:", collated_results)
    return collated_results
