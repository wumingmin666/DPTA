# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import os
import copy
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.structures import OptSampleList, SampleList, DetDataSample
from mmdet.models.utils import filter_scores_and_topk
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        # print(batch_data_samples)
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        #wmm
        self.text_feats, _ = self.backbone.forward_text(texts)
        # self.text_feats, None = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, (txt_feats,
                        txt_masks) = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats, txt_masks


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class YOLOWorldDetectorWithEnergy(YOLOWorldDetector):
    """
    在训练时将图能量（energy）加入总 loss。
    配置项示例（添加到模型 cfg）:
      use_energy: True
      energy_cfg: {graph_path: ..., size_stats_path: ..., angle_graph_path: ..., weight: 1.0}
    """
    def __init__(self,
                 *args,
                 use_energy: bool = False,
                 energy_cfg: dict = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_energy = use_energy
        self.energy_cfg = energy_cfg if energy_cfg is not None else {}
        self.energy_calculator = None
        self.local_iter = 0

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses and add graph energy when `use_energy` 为 True."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)

        if not self.use_energy:
            return losses

        # Warmup: 在最开始的几次迭代时不计算 energy
        self.local_iter += 1
        start_iter = self.energy_cfg.get('start_iter', 0)
        if self.local_iter < start_iter:
            return losses

        # lazy init energy calculator（延迟初始化，避免在构造时就加载大文件）
        if self.energy_calculator is None:
            cfg = self.energy_cfg
            self.energy_calculator = GraphEnergyLogPlus(
                graph_path=cfg.get('graph_path'),
                size_stats_path=cfg.get('size_stats_path'),
                angle_graph_path=cfg.get('angle_graph_path'),
                aspect_ratio_loss_weight=cfg.get('aspect_ratio_loss_weight', 1.0),
                size_loss_weight=cfg.get('size_loss_weight', 0.2),
                diou_loss_weight=cfg.get('diou_loss_weight', 0.2),
                angle_loss_weight=cfg.get('angle_loss_weight', 0.2))

        # ===== 获取 head 的原始输出（feature-level），然后用 predict_by_feat 生成 InstanceData 列表 =====
        # 说明：训练时一般不直接调用 predict（会依赖完整 metainfo），因此这里使用更底层的 predict_by_feat
        outs = self.bbox_head(img_feats, txt_feats, txt_masks)

        # ===== 构造 batch_img_metas（优先使用已有 metainfo，否则尝试从 data_samples 或 batch_inputs 推断） =====
        # 目标：确保每张图有一个 dict，至少包含 'ori_shape' 用于恢复 bbox 到原始图像坐标
        batch_img_metas = None
        batch_img_metas = batch_data_samples['img_metainfos']
        
        # 使用 predict_by_feat 生成预测实例（与 head 的 predict 接口不同，这里不依赖高层 DataSample）
        # wmm: 调整为不使用 NMS，以免截断梯度，同时限制数量防止 OOM
        # 使用 Top-K (nms_pre) 筛选置信度最高的框，禁用 NMS (with_nms=False)
        proposal_cfg = ConfigDict(
            nms_pre=self.energy_cfg.get('nms_pre', 2000),  # 限制参与计算的框数量，防止 OOM
            score_thr=self.energy_cfg.get('score_thr', 0.0), # 降低阈值，确保训练初期有框参与
            min_bbox_size=0,
            nms=dict(type='nms', iou_threshold=1.0), # 这里的配置可能被 with_nms=False 覆盖，但保留以防万一
            max_per_img=self.energy_cfg.get('max_per_img', 2000),
            multi_label=False
        )
        preds = self._predict_for_energy_loss(cls_scores=outs[0],
                                              bbox_preds=outs[1],
                                              batch_img_metas=batch_img_metas,
                                              cfg=proposal_cfg)
        # print(preds)
        # 筛选每个类别置信度最高的框 (Top-1 Per Class)
        filtered_preds = []
        for p in preds:
            if len(p) == 0:
                filtered_preds.append(p)
                continue
            
            # p.labels: (N,), p.scores: (N,)
            unique_labels = torch.unique(p.labels)
            keep_indices = []
            for label in unique_labels:
                # Find index of max score for this label
                mask = (p.labels == label)
                # Get indices where mask is True
                candidate_indices = torch.nonzero(mask, as_tuple=True)[0]
                # Get scores for these candidates
                candidate_scores = p.scores[candidate_indices]
                # Find argmax within candidates
                best_idx_in_candidates = torch.argmax(candidate_scores)
                # Map back to original index
                best_idx = candidate_indices[best_idx_in_candidates]
                keep_indices.append(best_idx)
            
            if len(keep_indices) > 0:
                keep_indices = torch.stack(keep_indices)
                filtered_preds.append(p[keep_indices])
            else:
                filtered_preds.append(p[[]])

        # 计算能量并加入 losses（可配置权重）..
        energy = self.energy_calculator.compute_batch_energy(filtered_preds, batch_img_metas)
        weight = self.energy_cfg.get('weight', 0.05)
        losses['loss_energy'] = energy * weight
        return losses

    def _predict_for_energy_loss(self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 batch_img_metas: List[dict],
                                 cfg: ConfigDict) -> List[InstanceData]:
        """A simplified, training-safe version of `predict_by_feat`.

        This method generates predictions for calculating training-time losses
        like graph energy. It avoids in-place operations, NMS, and rescaling
        to ensure that the gradient flow is not interrupted.

        Args:
            cls_scores (List[Tensor]): Box scores for each scale level.
            bbox_preds (List[Tensor]): Box energies / deltas for each scale
                level.
            batch_img_metas (List[dict]): Meta information of each image.
            cfg (ConfigDict): Test / postprocessing configuration.

        Returns:
            List[InstanceData]: Object detection results of each image.
        """
        head = self.bbox_head
        cfg = copy.deepcopy(cfg)
        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # Use cached priors from the head, or generate new ones if needed
        if featmap_sizes != head.featmap_sizes:
            head.mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            head.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(head.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * head.num_base_priors, ), stride)
            for featmap_size, stride in zip(featmap_sizes,
                                            head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # Flatten predictions
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  head.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = head.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        results_list = []
        for (bboxes, scores,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              batch_img_metas):

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores.flatten()
                empty_results.labels = scores.flatten().int()
                results_list.append(empty_results)
                continue

            # Use config for pre-NMS filtering
            score_thr = cfg.get('score_thr', 0.0)
            nms_pre = cfg.get('nms_pre', 100000)
            multi_label = cfg.get('multi_label', False)

            if not multi_label:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])
            results_list.append(results)

        return results_list
