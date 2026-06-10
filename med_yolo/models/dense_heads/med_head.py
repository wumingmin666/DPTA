import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS
from yolo_world.models.dense_heads import YOLOWorldHead
from mmdet.utils import reduce_mean
from mmcv.ops import RoIAlign
from mmyolo.models.utils import gt_instances_preprocess
from mmdet.structures.bbox import bbox_overlaps
import copy
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

@MODELS.register_module()
class UltrasoundYOLOHead(YOLOWorldHead):

    def __init__(self,
                 prototype_path: str,
                 loss_iou_branch=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_contrastive=dict(type='PrototypeContrastiveLoss', loss_weight=0.5),
                 loss_topology=dict(type='TopologyEnergyLoss', loss_weight=1.0),
                 # 特征维度配置
                 geo_input_dim=4,
                 geo_embed_dim=64,
                 vis_input_dim=512,  # 确保与 Backbone/Neck 输出一致
                 final_embed_dim=64,
                 alpha=1.0, beta=1.0, gamma=0.5,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # -----------------------------------------------------------
        # 1. 创新模块：IoU 预测分支
        # -----------------------------------------------------------
        # reg_max 默认为 16，4个坐标，所以输入是 64
        dist_channels = 4 * 16 
        
        # 极简设计：只用两层 1x1 卷积，学习从"分布形状"到"IoU分数"的映射
        # 这种设计参数量极少，非常适合 10-Shot
        self.iou_predictor = nn.Sequential(
            nn.Conv2d(dist_channels, 32, 1), # 压缩特征
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),             # 输出 Score
            nn.Sigmoid()
        )
        """
        # 伪代码：直接用分布的确定性作为 IoU Score
        # 不需要 IoU Predictor 网络
        probs = dist_pred.softmax(dim=2) # [B, 4, 16, H, W]
        max_probs, _ = probs.max(dim=2)  # [B, 4, H, W] 取每个坐标的最高置信度
        uncertainty_score = max_probs.mean(dim=1, keepdim=True) # [B, 1, H, W] 平均置信度
        iou_preds.append(uncertainty_score)
        
        """
        # 别忘了 Bias 初始化策略 (策略 A)
        nn.init.constant_(self.iou_predictor[-2].bias, 0.405)
          

        # -----------------------------------------------------------
        # 2. 创新模块：Metric 分支组件 (适配 512维)
        # -----------------------------------------------------------
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=0)
        
        # 几何升维 (4 -> 64)
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, geo_embed_dim),
            nn.LayerNorm(geo_embed_dim),
            nn.ReLU()
        )
        
        # 融合层 (512 + 64 -> 64)
        self.fusion_input_dim = vis_input_dim + geo_embed_dim 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, final_embed_dim)
        )
        
        # 加载 Loss 和 原型
        self.loss_iou_branch = MODELS.build(loss_iou_branch)
        self.loss_con = MODELS.build(loss_contrastive)
        self.loss_energy = MODELS.build(loss_topology)
        self.prototypes = self._load_prototypes(prototype_path)


        self.alpha = alpha  # Text Score 权重
        self.beta = beta    # IoU Score 权重
        self.gamma = gamma  # Similarity Score 权重

    def _load_prototypes(self, path):
        """
        加载原型库，适配用户保存的字典结构:
        Keys: ['class_ids', 'vis_prototypes', 'geo_prototypes']
        """
        # 加载 .pt 文件
        data = torch.load(path, map_location='cpu')
        
        # 1. 维度检查
        # visual prototypes shape: [N, 512]
        vis_dim = data['vis_prototypes'].shape[1]
        geo_dim = data['geo_prototypes'].shape[1]
        
        assert vis_dim == 512, f"Visual prototype dimension mismatch! Expected 512, got {vis_dim}"
        assert geo_dim == 4, f"Geometric prototype dimension mismatch! Expected 4, got {geo_dim}"
        
        # 2. 注册为 Buffer (随模型移动到 GPU，不更新参数)
        # 注意：这里我们修改了键名以匹配你的保存代码
        self.register_buffer('proto_vis', data['vis_prototypes'])  # [6, 512]
        self.register_buffer('proto_geo', data['geo_prototypes'])  # [6, 4]
        self.register_buffer('proto_ids', data['class_ids'])       # [6]
        
        # 3. (可选) 打印加载信息
        print(f"[UltrasoundHead] Loaded prototypes: {len(data['class_ids'])} classes.")
        print(f" - Visual Shape: {self.proto_vis.shape}")
        print(f" - Geo Shape:    {self.proto_geo.shape}")
        
        return data

    def forward(self, img_feats, txt_feats, txt_masks):
        """
        重写 Forward，拦截 img_feats 用于 IoU 分支计算
        """
        # 1. 调用父类 forward 获取标准输出
        # parent returns: (cls_logits, bbox_preds, bbox_dist_preds)
        # outs[2] 就是 bbox_dist_preds
        outs = super().forward(img_feats, txt_feats, txt_masks) #调用YOLOWorldHeadModule去下采样
        
        iou_preds = []
        bbox_dist_preds_list = outs[2]
        # 使用 zip 同时遍历 特征图(为了拿H,W) 和 分布预测
        for feat, dist_pred in zip(img_feats, bbox_dist_preds_list):
            # feat shape: [B, C, H, W]
            # dist_pred shape: [B, H*W, 4, 16] (这是导致报错的形状)
                
            B, C, H, W = feat.shape
                
            # 1. 截断梯度 (不影响主干)
            dist_feat = dist_pred.detach()
                
            # 2. 【关键修复】维度还原
            # 目前: [B, H*W, 4, 16]
            # 目标: [B, 64, H, W] (其中 64 = 4 * 16)
                
            # Permute: [B, H*W, 4, 16] -> [B, 4, 16, H*W]
            dist_feat = dist_feat.permute(0, 2, 3, 1)
                
            # Merge Channels: [B, 4, 16, H*W] -> [B, 64, H*W]
            dist_feat = dist_feat.reshape(B, -1, H * W)
                
            # Reshape Spatial: [B, 64, H*W] -> [B, 64, H, W]
            dist_feat = dist_feat.reshape(B, -1, H, W)
                
            # 3. 现在形状正确了，可以传入 Conv2d
            iou_pred = self.iou_predictor(dist_feat)
            iou_preds.append(iou_pred)   
        # 仅在训练模式下 outs 才有 3 个元素 (dist_preds)
        if self.training:
            return outs[0], outs[1], outs[2], iou_preds
        else:
            # 推理模式下逻辑 (根据之前的实现)
            return outs[0], outs[1], iou_preds

    def loss(self, img_feats, txt_feats, txt_masks, batch_data_samples):
        """
        重写 loss，确保参数传递顺序与 loss_by_feat 严格一致
        """
        # 1. 前向传播 (返回 cls, bbox, dist, iou)
        outs = self(img_feats, txt_feats, txt_masks)
        
        # 2. 构造输入元组 (Strict Order)
        loss_inputs = (
            img_feats,                          # 1. 特征图
        ) + outs + (                            # 2,3,4,5. 网络输出 (cls, bbox, dist, iou)
            batch_data_samples['bboxes_labels'],# 6. GT 实例
            batch_data_samples['img_metas'],    # 7. 图片元信息
            txt_masks                           # 8. [修复] 补上文本掩码
        )
        
        # 3. 计算损失
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self, 
                     img_feats,
                     cls_scores, 
                     bbox_preds, 
                     bbox_dist_preds,    # [修复] 加上了 bbox_dist_preds
                     iou_preds,          # [新增] 我们的 IoU 分支输出
                     batch_gt_instances, 
                     batch_img_metas,
                     batch_text_masks=None,
                     batch_gt_instances_ignore=None):
        # -----------------------------------------------------------
        # 验证逻辑 (可选，用于 Debug)
        # -----------------------------------------------------------
        # print(f"DEBUG: img_feats type: {type(img_feats)}") # Should be tuple/list
        # print(f"DEBUG: iou_preds len: {len(iou_preds)}")   # Should be 3 (levels)
        # print(f"DEBUG: txt_masks: {batch_text_masks is not None}")
        num_imgs = len(batch_img_metas)
        device = cls_scores[0].device

        # ==================================================================
        # PART 1: 官方 YOLO-World 预处理逻辑 (原样复刻)
        # ==================================================================
        
        # 1.1 动态 Anchor 生成
        current_featmap_sizes = [c.shape[2:] for c in cls_scores]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train, dtype=cls_scores[0].dtype,
                device=device, with_stride=True)
            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # 1.2 GT 预处理
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # 1.3 Flatten 预测值
        flatten_cls_preds = [
            c.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for c in cls_scores
        ]
        flatten_pred_bboxes = [
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for b in bbox_preds
        ]
        # [修复] 处理 bbox_dist_preds (用于 DFL)
        flatten_pred_dists = [
            d.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for d in bbox_dist_preds
        ]
        # [新增] Flatten IoU Preds
        flatten_iou_preds = [
            i.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for i in iou_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_iou_preds = torch.cat(flatten_iou_preds, dim=1)

        # 1.4 解码 (Decode)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        # 1.5 正负样本分配 (Assigner)
        assigned_result = self.assigner(
            (flatten_decoded_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), 
            self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # print(assigned_result)
        # print(assigned_result['assigned_labels'][0].shape)
        # print(assigned_result['assigned_labels'][0])
        # ==================================================================
        # PART 2: 官方 Loss 计算 (Cls + Bbox + DFL)
        # ==================================================================
        
        # 2.1 Classification Loss (含 Text Mask 逻辑)
        # print(flatten_cls_preds.shape)
        if batch_text_masks is not None:
            cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
                -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            loss_cls = (loss_cls * cls_weight).sum()
        else:
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # **重要提示**：在归一化之前，备份一份绝对坐标用于 Topology Loss
        # 因为 loss_bbox 计算通常会执行 /= stride，破坏绝对坐标
        decoded_bboxes_for_topology = flatten_decoded_bboxes.clone() 

        # 2.2 Bbox Loss & DFL Loss
        # 归一化预测框
        assigned_bboxes /= self.stride_tensor
        flatten_decoded_bboxes /= self.stride_tensor

        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_decoded_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            
            # IoU Loss (CIoU)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # [修复] DFL Loss (官方逻辑)
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1, eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0

        # ==================================================================
        # PART 3: 你的创新 Loss 计算 (IoU Branch + Metric + Topology)
        # ==================================================================

        # 3.1 IoU 预测分支 Loss
        if num_pos > 0:
            pos_iou_preds = flatten_iou_preds[fg_mask_pre_prior].squeeze(-1)
            # 使用 DFL 计算前的绝对坐标计算 IoU Target
            # 重新解码正样本用于计算 target (或者使用 clone 的那份)
            # 这里为了简单，直接用已经选出来的 pos_bboxes (注意它们已经被除以 stride 了，所以要一致)
            # 最稳妥是使用解码后的绝对坐标计算 IoU

            # --- [修复开始] ---
            # 1. 获取 Batch Size
            batch_size = fg_mask_pre_prior.size(0)
            
            # 2. 将 stride_tensor 从 [N, 1] 扩展为 [B, N, 1]
            # 这样它就和 fg_mask_pre_prior [B, N] 维度对齐了
            batch_stride_tensor = self.stride_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 3. 现在可以安全地使用掩码提取正样本的 stride
            pos_stride = batch_stride_tensor[fg_mask_pre_prior].squeeze(-1)
            # --- [修复结束] ---

            # 从 backup 中恢复绝对坐标的正样本
            pos_abs_pred = decoded_bboxes_for_topology[fg_mask_pre_prior]
            # 恢复 GT 的绝对坐标 (assigned_bboxes 已经被除了，所以我们需要 assigned_result 里的原始信息)
            # 更好的办法：直接重新取 gt_bboxes
            # 这里我们利用 assigned_bboxes * stride 还原回去
            # pos_stride = self.stride_tensor[fg_mask_pre_prior].squeeze(-1)
            pos_abs_gt = assigned_bboxes_pos * pos_stride.unsqueeze(1)

            # iou_targets = self.bbox_coder.iou_calculator(
            #     pos_abs_pred, pos_abs_gt).clamp(0, 1)
            
            # [修复] 使用 bbox_overlaps 替代 self.bbox_coder.iou_calculator
            iou_targets = bbox_overlaps(
                pos_abs_pred, 
                pos_abs_gt, 
                is_aligned=True
            ).clamp(0, 1)
            
            loss_iou_branch = self.loss_iou_branch(pos_iou_preds, iou_targets.detach())
        else:
            loss_iou_branch = flatten_iou_preds.sum() * 0

        # 3.2 准备 Metric 和 Topology 数据
        # 需要找到正样本属于哪张图，以便做 RoIAlign 和 Image-wise Topology
        # num_pos_per_img = [res.fg_mask_pre_prior.sum().item() for res in [assigned_result]] 
        # 1. 从字典中通过 Key 获取掩码
        # fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        
        # 2. 计算每张图的正样本数量
        # fg_mask_pre_prior shape: [Batch_Size, Total_Anchors]
        # dim=1 求和 -> [Batch_Size] -> 转为 list
        # num_pos_per_img = fg_mask_pre_prior.sum(dim=1).int().tolist()

        # 注意: assigned_result 返回的是合并后的 dict，我们需要手动拆分 mask
        # 修正：mmyolo 的 assigner 返回的 fg_mask_pre_prior 已经是 [Batch, Total_Priors]
        # 所以我们需要按 batch 维度拆分
        fg_mask_per_img = fg_mask_pre_prior.view(num_imgs, -1)
        num_pos_per_img = fg_mask_per_img.sum(dim=1).int().tolist()

        # 构建 Batch Index 用于 RoI Align
        batch_inds_list = []
        for i, n in enumerate(num_pos_per_img):
            batch_inds_list.append(torch.full((n, 1), i, device=device))
        
        if num_pos > 0:
            batch_inds = torch.cat(batch_inds_list, dim=0)
            # 使用备份的绝对坐标
            pos_abs_pred = decoded_bboxes_for_topology[fg_mask_pre_prior]
            rois = torch.cat([batch_inds, pos_abs_pred], dim=1) # [N, 5]

            # --- Metric Branch (Contrastive) ---
            # A. 视觉特征 (512维)
            """
            Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
            """
            # 2. 选择特征层
            # img_feats 是一个 tuple (P3, P4, P5)
            # 我们选择最后一层 P5 (通常语义最强，且维度可能为 512)
            target_feat = img_feats[2] 
            
            # [重要] 动态调整 spatial_scale
            # 因为 RoI 是绝对坐标，而 feature map 是下采样的
            # P5 的 stride 是 32 (640 -> 20)
            # 你可以在 __init__ 里固定 self.roi_align = RoIAlign(..., spatial_scale=1/32)
            # 或者在这里临时处理 (如果你的 RoIAlign spatial_scale 是 1.0)
            
            # 建议方案：直接在这里缩放 ROI，而不是改 RoIAlign 的参数（比较灵活）
            stride = 32.0 
            rois_rescaled = rois.clone()
            rois_rescaled[:, 1:] /= stride # 将坐标映射到特征图尺度

            # 3. 执行 RoI Align
            # 输入: Tensor [B, 512, H, W]
            # 输出: Tensor [N, 512, 7, 7]
            roi_feats = self.roi_align(target_feat, rois_rescaled)

            # print(roi_feats.shape)


            # roi_feats = self.roi_align(cls_scores, rois) # [N, 512, 7, 7]
            f_vis = roi_feats.mean(dim=[2, 3]) # [N, 512]

            # B. 几何特征 (4维 -> 64维)
            w = (pos_abs_pred[:, 2] - pos_abs_pred[:, 0]).unsqueeze(1)
            h = (pos_abs_pred[:, 3] - pos_abs_pred[:, 1]).unsqueeze(1)

            # print(batch_img_metas)
            img_h, img_w = batch_img_metas[0]['batch_input_shape'][:2]
            f_geo_raw = torch.cat([w/img_w, h/img_h, w/h, w*h/(img_w*img_h)], dim=1)
            f_geo_embed = self.geo_mlp(f_geo_raw)

            # C. 融合与计算
            query_embed = self.fusion_mlp(torch.cat([f_vis, f_geo_embed], dim=1))
            
            # D. 原型 Key 计算
            proto_geo_embed = self.geo_mlp(self.proto_geo)
            key_embed = self.fusion_mlp(torch.cat([self.proto_vis, proto_geo_embed], dim=1))
            
            # 获取对应的 Label
            # assigned_result['assigned_labels'] 包含所有 anchor 的 label
            # 我们需要正样本的 label
            # mmyolo assigner 输出通常不直接包含 assigned_labels，通常需要利用 gt_labels 和 mask
            # 简单方法：利用 mmyolo 的 assigned_labels 属性如果存在，或者重新通过 gt_info 获取
            # 这里简化逻辑：我们复用 assigner 内部逻辑通常会产生的 assigned_labels
            # 如果没有，我们可以用 gt_labels 和 assigned_gt_inds 获取
            # 假设 assigner 结果里有 'assigned_labels' (MMYOLO TaskAlignedAssigner 通常有)
            # print("====================================")
            # print(assigned_result)
            pos_labels = assigned_result['assigned_labels'][fg_mask_pre_prior]
            
            loss_con = self.loss_con(query_embed, key_embed, pos_labels)
        else:
            loss_con = flatten_iou_preds.sum() * 0

        # 3.3 Topology Energy Loss (Image-wise)
        loss_energy = 0.0
        start_idx = 0
        
        # 获取所有正样本对应的 GT (绝对坐标)
        # 我们需要从 assigned_bboxes (已除 stride) 恢复，或者更准确地，直接从 gt_bboxes 获取
        # 为了精度，我们直接使用 assigner 知道的 gt index
        # mmyolo 的 assigner 通常返回 'assigned_gt_inds' (1-based, 0 is ignore)
        # print(assigned_result)
        # assigned_gt_inds = assigned_result['assigned_gt_inds'][fg_mask_pre_prior]
        all_assigned_bboxes = assigned_result['assigned_bboxes']
        for i, n in enumerate(num_pos_per_img):
            if n < 2: 
                start_idx += n
                continue
            
            # 当前图片的正样本预测框 (绝对坐标)
            img_pred_bboxes = decoded_bboxes_for_topology[i][fg_mask_per_img[i]]
            
            # 当前图片的正样本对应的 GT (绝对坐标)
            # 注意：assigned_gt_inds 是全局 flatten 后的还是 per image 的？
            # TaskAlignedAssigner 的结果通常是 [Batch, Anchors]。
            # 我们这里取出了当前图的正样本 GT 索引 (1-based)
            img_gt_relative = all_assigned_bboxes[i][fg_mask_pre_prior[i]]
            img_stride = self.stride_tensor[fg_mask_pre_prior[i]]
            img_gt_bboxes = img_gt_relative * img_stride

            # img_gt_inds = assigned_result['assigned_gt_inds'][i][fg_mask_per_img[i]] - 1
            # img_gt_bboxes = gt_bboxes[i][img_gt_inds] # 直接索引原始 GT

            loss_energy += self.loss_energy(img_pred_bboxes, img_gt_bboxes)
            start_idx += n
            
        loss_energy /= num_imgs

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            loss_iou_branch=loss_iou_branch,
            loss_con=loss_con,
            loss_energy=loss_energy
        )
    
    def predict(self,
                img_feats,
                txt_feats,
                txt_masks,
                batch_data_samples,
                rescale=False):
        """
        重写 predict，以便将 img_feats 透传给 predict_by_feat
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        
        # 1. 获取 Head 输出 (包含 iou_preds)
        # outs = (cls_scores, bbox_preds, iou_preds) 
        # 注意：forward 在非 training 模式下返回的是这个三元组
        outs = self(img_feats, txt_feats, txt_masks)
        
        # 2. 调用 predict_by_feat
        # 这里我们可以利用 python 的闭包特性或者直接修改 predict_by_feat 的签名
        # 最简单的方法：把 img_feats 暂时绑定到 self 上 (虽然不推荐多线程，但推理通常没问题)
        self.temp_img_feats = img_feats 
        # print(outs)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        
        # 清理
        self.temp_img_feats = None
        
        return predictions
    # ==================================================================
    # 新增：推理配置 (在 __init__ 中添加)
    # ==================================================================
    # def __init__(self, ..., alpha=1.0, beta=1.0, gamma=0.5, **kwargs):
    #     self.alpha = alpha  # Text Score 权重
    #     self.beta = beta    # IoU Score 权重
    #     self.gamma = gamma  # Similarity Score 权重
    #     ...

    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        iou_preds,       # 我们自定义的 IoU 分支
                        batch_img_metas,
                        cfg=None,
                        rescale=True,
                        with_nms=True):
        """
        实现方案中的推理逻辑：Forward -> Calibration -> Fusion -> NMS
        """
        assert len(cls_scores) == len(bbox_preds)
        
        # 1. 准备配置
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # 2. 生成 Anchors (Priors)
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)
        
        # 3. 展平所有层的预测 (Flatten)
        # cls_scores: [B, N, C]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        # bbox_preds: [B, N, 4]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # iou_preds: [B, N, 1] (自定义分支)
        flatten_iou_preds = [
            iou_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for iou_pred in iou_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_iou_preds = torch.cat(flatten_iou_preds, dim=1).sigmoid() # IoU Score
        # print(flatten_bbox_preds.shape)
        # print(flatten_priors.shape)
        # print(self.stride_tensor.shape)
        # 4. 解码 BBox (Decode)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, self.stride_tensor.flatten())
        # print(flatten_decoded_bboxes.shape)
        # flatten_decoded_bboxes = flatten_decoded_bboxes.reshape(num_imgs, -1, 4)
        # ===========================================================
        # 核心推理循环 (Image-wise)
        # ===========================================================
        results_list = []
        for img_id in range(num_imgs):
            score_thr = cfg.get('score_thr', 0.05) # 初筛阈值
            
            # --- A. 获取当前图的数据 ---
            cls_score = flatten_cls_scores[img_id]      # [N, C] (S_text)
            bboxes = flatten_decoded_bboxes[img_id]     # [N, 4]
            iou_score = flatten_iou_preds[img_id]       # [N, 1] (S_iou)
            img_meta = batch_img_metas[img_id]
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:#进入
                pad_param = img_meta['pad_param']
            else:
                pad_param = None
            # --- B. 初步过滤 (Preliminary Filtering)  ---
            # 如果不先过滤，对 8400 个框做 RoIAlign 速度会极其慢
            # 使用 Text Score 的最大值进行筛选
            max_scores, _ = cls_score.max(dim=1)
            valid_mask = max_scores > score_thr
            
            # 如果没有框通过初筛，直接返回空
            if valid_mask.sum() == 0:
                results_list.append(InstanceData(
                    bboxes=torch.empty((0, 4), device=bboxes.device),
                    scores=torch.empty((0,), device=bboxes.device),
                    labels=torch.empty((0,), device=bboxes.device)))
                continue

            # 筛选后的数据
            # print("===",bboxes.shape)
            valid_bboxes = bboxes[valid_mask]       # [M, 4]
            valid_cls = cls_score[valid_mask]       # [M, C]
            valid_iou = iou_score[valid_mask]       # [M, 1]

            # --- C. 原型校准 (Prototype Calibration)  ---
            # 计算 sim_score (S_sim)
            # 1. 构造 RoIs: [0, x1, y1, x2, y2] (batch_id 全为 0，因为我们现在是单张处理)
            batch_inds = valid_bboxes.new_zeros((valid_bboxes.size(0), 1))
            # rois = torch.cat([batch_inds, valid_bboxes], dim=1)
            
            # 2. 提取特征 (Vis + Geo)
            # 注意：roi_align 需要 list of feats，这里我们要把 neck feats 传进来很难
            # 替代方案：在 forward 里把 img_feats 存为 self.current_feats (虽不优雅但可行)
            # 或者：简化实现，暂时假设 self.current_feats 可用
            # **最佳实践**：我们实际上不能在 head 存状态。
            # 为了解决这个问题，我们需要在 predict 阶段传入 img_feats。
            # (下文会解决这个问题，这里假设 feats 可用)
            
            # ... (假设 roi_feats 已提取) ...
            # 此处为了演示逻辑完整性，伪代码表示 Metric 计算：
            # sim_scores = self._calculate_metric_scores(valid_bboxes, self.current_feats) 
            
            # 在无法获取特征的情况下（标准接口限制），我们可以暂时跳过 Sim 分支
            # 或者修改 Detector 传入 Feats。
            # 这里假设 sim_scores 默认为 1.0 (如果无法提取特征)
            # sim_scores = torch.ones_like(valid_iou) 
            

            # --- 补全：C. 原型校准 ---
            # 从 self.temp_img_feats 获取特征
            if hasattr(self, 'temp_img_feats') and self.temp_img_feats is not None:
                # RoI Align
                # print(valid_bboxes.shape)
                rois = torch.cat([batch_inds, valid_bboxes], dim=1) # [M, 5]
                # roi_feats = self.roi_align(self.temp_img_feats, rois)
                target_feat = self.temp_img_feats[2][img_id:img_id+1]
                stride = 32.0
                rois_rescaled = rois.clone()
                rois_rescaled[:, 1:] /= stride
                roi_feats = self.roi_align(target_feat, rois_rescaled)
                
                # 提取 Vis 特征
                f_vis = roi_feats.mean(dim=[2, 3]) # [M, 512]
                
                # 提取 Geo 特征
                w = (valid_bboxes[:, 2] - valid_bboxes[:, 0]).unsqueeze(1)
                h = (valid_bboxes[:, 3] - valid_bboxes[:, 1]).unsqueeze(1)
                img_h, img_w = img_meta['img_shape'][:2]
                f_geo_raw = torch.cat([w/img_w, h/img_h, w/h, w*h/(img_w*img_h)], dim=1)
                f_geo_embed = self.geo_mlp(f_geo_raw)
                
                # 计算 Query Embedding
                query_embed = self.fusion_mlp(torch.cat([f_vis, f_geo_embed], dim=1))
                
                # 计算 Key Embedding (Prototype)
                proto_geo_embed = self.geo_mlp(self.proto_geo)
                key_embed = self.fusion_mlp(torch.cat([self.proto_vis, proto_geo_embed], dim=1))
                
                # 计算 Cosine Similarity [M, C]
                # query: [M, 64], key: [C, 64]
                sim_matrix = torch.matmul(
                    F.normalize(query_embed, p=2, dim=1),
                    F.normalize(key_embed, p=2, dim=1).t()
                )
                
                # sim_matrix[i, c] 表示第 i 个框属于 c 类的相似度
                # 我们需要让这个相似度与 valid_cls 对应
                # valid_cls 已经是 [M, C] 的分数了
                # 我们可以直接把 sim_matrix 作为一个系数乘上去
                sim_scores = sim_matrix.clamp(min=0) # 相似度截断在 [0, 1]
                
            else:
                sim_scores = 1.0
            # --- D. 分数融合 (Score Fusion)  ---
            # S_final = S_text^alpha * S_iou^beta * S_sim^gamma
            # 为避免数值过小，通常在 log 域加权或者直接乘
            
            # 融合公式：
            # S_final = S_text * (S_iou ^ beta) * (S_sim ^ gamma)
            # 注意：S_text 已经包含在 valid_cls 里了
            
            fusion_factor = (valid_iou ** self.beta) * (sim_scores ** self.gamma)
            final_scores = valid_cls * fusion_factor # 广播机制 [M, C] * [M, 1]

            # --- E. 最终 NMS (Global NMS)  ---
            # 转换为 (x1, y1, x2, y2, score) 格式供 NMS 使用
            # --- [修复] 1. 打包输入数据 ---
            results = InstanceData()
            results.bboxes = valid_bboxes
            # results.scores = final_scores

            # # [修复] 添加 labels
            # if final_scores.shape[0] > 0:
            #     results.labels = final_scores.argmax(dim=1)
            # else:
            #     results.labels = final_scores.new_zeros(0, dtype=torch.long)

            if final_scores.shape[0] > 0:
                # 1. 获取每个框概率最大的类别索引 [N]
                labels = final_scores.argmax(dim=1)
                
                # 2. 获取对应的分数值 [N]
                # 使用 gather 或者高级索引提取对应类别的分数
                # scores 必须是 1D 张量，否则 NMS 会报错
                scores = final_scores[torch.arange(final_scores.size(0)), labels]
                
                results.labels = labels
                results.scores = scores
            else:
                results.labels = final_scores.new_zeros(0, dtype=torch.long)
                results.scores = final_scores.new_zeros(0)


            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            det_results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                img_meta=img_meta  # 指定关键字参数
            )
            # --- [修复] 3. 解析返回结果 ---
            # det_results 里面包含了经过 NMS 后的 bboxes, scores, labels
            
            # 将结果存入 results_list
            # 注意：mmyolo 的 predict_by_feat 通常直接返回 InstanceData 列表
            # 不需要像之前那样手动拆包再封装
            results_list.append(det_results)
            # det_bboxes, det_labels = self._bbox_post_process(
            #     valid_bboxes, final_scores, cfg, img_meta, rescale=rescale)
            
            # --- F. (可选) 能量检查 [cite: 152] ---
            # 如果需要，可以在这里对 det_bboxes 再次进行基于 L_energy 的过滤
            # 但通常 NMS 已经足够

            # results = InstanceData()
            # results.bboxes = det_bboxes[:, :4]
            # results.scores = det_bboxes[:, 4]
            # results.labels = det_labels
            # results_list.append(results)

        return results_list

@MODELS.register_module()
class MedYOLOWorldHead(YOLOWorldHead):
    """
    Anatomy-Aware & Prototype-Calibrated Head for Ultrasound Few-Shot Detection.
    Inherits from YOLOWorldHead to reuse the text-image interaction capabilities.
    """
    def __init__(self,
                 prototype_path: str,
                 loss_iou_branch=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_contrastive=dict(type='PrototypeContrastiveLoss', loss_weight=0.5),
                 loss_topology=dict(type='TopologyEnergyLoss', loss_weight=1.0),
                 proto_mlp_cfg=dict(hidden_dim=128, out_dim=64),
                 geo_input_dim=4,     # 原始几何特征维度 [w, h, r, a]
                 geo_embed_dim=64,    # 几何特征升维后的维度
                 vis_input_dim=512,   # 视觉特征维度
                 final_embed_dim=64,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # -----------------------------------------------------------
        # [cite_start]1. 新增：IoU 预测分支 (Quality Estimator) [cite: 96-100]
        # 结构: Conv3x3 -> SiLU -> Conv1x1 -> Sigmoid
        # 输入通道数通常为 self.feat_channels (YOLOv8/World中通常是reg_max * 4 或者 hidden dim)
        # 注意：YOLO-World Head 的回归分支输出是 reg_max * 4，这里我们将 IoU 分支接在回归分支的特征图上
        # 但为了简单起见，我们通常接在 cls_preds 或 reg_preds 之前的 stem 上。
        # 这里假设接在 Regression Head 的共享卷积之后 (reg_convs)。
        # -----------------------------------------------------------
        self.iou_predictor = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.feat_channels, 1, 1),
            nn.Sigmoid()
        )

        # -----------------------------------------------------------
        # [cite_start]2. 新增：原型 Metric 分支组件 [cite: 102-113]
        # -----------------------------------------------------------
        # RoI Align 提取器 (输出 7x7)
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=0)

        # 1. 几何特征升维器 (Geo Encoder)
        # 作用：将 4维 -> 64维，使其能与视觉特征抗衡
        

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, geo_embed_dim),
            nn.LayerNorm(geo_embed_dim),
            nn.ReLU()
        )

        # 2. 融合投影层 (Fusion Projector)
        # 作用：将 (512视觉 + 64几何) -> 64最终Embedding
        # 这一层让两种特征真正交互
        self.fusion_input_dim = vis_input_dim + geo_embed_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 256), # 中间层适当放大以承载 576 维输入
            nn.LayerNorm(256),                     # 加个 LayerNorm 会更稳
            nn.ReLU(),
            nn.Linear(256, final_embed_dim)        # Output: 64
        )

        
        # # 孪生 MLP (Siamese MLP Adaptor)
        # # Input: 256 (Vis) + 4 (Geo) = 260
        # self.proto_input_dim = 256 + 4 
        # self.proto_mlp = nn.Sequential(
        #     nn.Linear(self.proto_input_dim, proto_mlp_cfg['hidden_dim']),
        #     nn.LayerNorm(proto_mlp_cfg['hidden_dim']),
        #     nn.ReLU(),
        #     nn.Linear(proto_mlp_cfg['hidden_dim'], proto_mlp_cfg['out_dim'])
        # )
        
        # 加载原型库 (Offline Stage 2 生成的 .pt 文件)
        self.prototypes = self._load_prototypes(prototype_path)
        
        # -----------------------------------------------------------
        # 3. 初始化自定义 Loss
        # -----------------------------------------------------------
        self.loss_iou_branch = MODELS.build(loss_iou_branch)
        self.loss_con = MODELS.build(loss_contrastive)
        self.loss_energy = MODELS.build(loss_topology)

    def _load_prototypes(self, path):
        """加载并注册原型为 Buffer (不更新但参与计算)"""
        data = torch.load(path, map_location='cpu')
        # 假设 data 包含 'vis_feats' [C, 256] 和 'geo_feats' [C, 4]
        # 我们需要 register_buffer 以便它们随模型移动到 GPU
        self.register_buffer('proto_vis', data['vis_feats'])
        self.register_buffer('proto_geo', data['geo_feats'])
        return data

    def forward(self, x):
        """
        前向传播需要同时输出 IoU 预测图。
        x: list of features from Neck (P3, P4, P5)
        """
        # 1. 调用父类 forward 获取标准输出
        # standard_outs 通常是 tuple(cls_scores, bbox_preds) 或类似结构
        # 在 YOLO-World 中，它可能返回 (cls_scores, bbox_preds, objectness) 视版本而定
        cls_scores, bbox_preds = super().forward(x)
        
        # 2. 计算 IoU 分支输出
        iou_preds = []
        # 注意：这里我们假设复用 Regression 分支的特征，或者直接从 Neck 特征 x 计算
        # 更加稳健的做法是直接使用 x，因为 reg_convs 是内部变量可能访问不到
        for feat in x:
            iou_preds.append(self.iou_predictor(feat))
            
        # 返回三元组，以便后续 loss_by_feat 接收
        return cls_scores, bbox_preds, iou_preds

    def loss_by_feat(self, cls_scores, bbox_preds, iou_preds,
                     batch_gt_instances, batch_img_metas,
                     batch_gt_instances_ignore=None):
        """
        计算所有 Loss。这里我们需要手动执行 Label Assignment，
        以便拿到正样本去计算 Metric Loss 和 Topology Loss。
        """
        
        # -----------------------------------------------------------
        # Step 1: 准备数据和 Anchor (Priors)
        # -----------------------------------------------------------
        device = cls_scores[0].device
        # 展平预测结果 (Flatten)
        flatten_cls_scores = torch.cat([
            c.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for c in cls_scores
        ], dim=0)
        flatten_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(-1, 4)
            for b in bbox_preds
        ], dim=0)
        flatten_iou_preds = torch.cat([
            i.permute(0, 2, 3, 1).reshape(-1, 1)
            for i in iou_preds
        ], dim=0)

        # 生成 Anchors / Priors (用于解码 bbox 和 Assignment)
        anchor_points, stride_tensor = self.prior_generator.generate_anchors(
            [c.shape[2:] for c in cls_scores], 
            [c.shape[1] for c in cls_scores], # featmap sizes
            device=device
        )
        
        # 解码所有预测框 (从 dist 变为 xyxy)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            anchor_points, flatten_bbox_preds, stride_tensor)

        # -----------------------------------------------------------
        # Step 2: 正负样本分配 (Label Assignment)
        # -----------------------------------------------------------
        assigned_results = self.assigner(
            (flatten_decoded_bboxes.detach(), flatten_cls_scores.detach(), anchor_points),
            batch_gt_instances,
            batch_gt_instances_ignore,
            batch_img_metas
        )
        
        # 解析 Assignment 结果
        # pos_mask: [Total_Anchors] 指示正样本位置
        # pos_gt_inds: [Total_Anchors] 指示正样本对应的 GT 索引
        # pos_bbox_targets: [Total_Pos, 4] 正样本对应的 GT 框
        pos_mask = torch.cat([res.pos_mask for res in assigned_results])
        pos_gt_labels = torch.cat([res.pos_gt_labels for res in assigned_results])
        pos_gt_bboxes = torch.cat([res.pos_gt_bboxes for res in assigned_results])
        
        # -----------------------------------------------------------
        # Step 3: 计算基础 Loss (Cls + Reg)
        # -----------------------------------------------------------
        # 复用父类方法太麻烦，直接调用 loss 模块
        # 注意：这里简化了计算，实际使用时建议参考 YOLOWorldHead 源码的 loss 计算部分
        # VFL / DFL / CIoU 计算 ... (为节省篇幅略过，假设你保留原有逻辑)
        # 这里重点展示新增的 Loss 计算
        
        # 为了演示，我们假设 loss_cls 和 loss_bbox 已经算好了 (你可以直接 copy 源码)
        # loss_cls = ...
        # loss_bbox = ...
        
        # -----------------------------------------------------------
        # [cite_start]Step 4: 计算 IoU 分支 Loss [cite: 123-126]
        # -----------------------------------------------------------
        # 只计算正样本
        if pos_mask.sum() > 0:
            pos_iou_preds = flatten_iou_preds[pos_mask].squeeze(-1)
            pos_decoded_bboxes = flatten_decoded_bboxes[pos_mask]
            
            # 计算 Target IoU (预测框 vs GT)
            iou_targets = self.bbox_coder.iou_calculator(
                pos_decoded_bboxes, pos_gt_bboxes).clamp(0, 1)
            
            loss_iou_branch = self.loss_iou_branch(pos_iou_preds, iou_targets.detach())
        else:
            loss_iou_branch = flatten_iou_preds.sum() * 0

        # -----------------------------------------------------------
        # [cite_start]Step 5: 准备 Metric Branch 特征 [cite: 108-111]
        # -----------------------------------------------------------
        # 我们需要对正样本位置提取特征。
        # 难点：flatten_decoded_bboxes 对应的特征在 Feature Map 里的哪个位置？
        # 简单方案：直接用 RoIAlign 提取 Feature Maps
        
        # 构建 RoIs: [Batch_ID, x1, y1, x2, y2]
        # 我们需要知道每个正样本属于哪张图
        num_pos_per_img = [res.pos_mask.sum().item() for res in assigned_results]
        batch_ids = []
        for i, num in enumerate(num_pos_per_img):
            batch_ids.append(torch.full((num, 1), i, device=device))
        
        if len(batch_ids) > 0 and pos_mask.sum() > 0:
            batch_ids = torch.cat(batch_ids, dim=0)
            rois = torch.cat([batch_ids, pos_decoded_bboxes], dim=1) # [N_pos, 5]
            
            # 5.1 提取视觉特征 F_vis
            # 注意：RoIAlign 需要 feature list。我们简单选取 P3 (分辨率最高) 或多尺度
            # 这里为演示简便，取 output_size=1x1 然后 flatten，或者 7x7 后 pooling
            # 实际上应该用 MultiScaleRoIAlign
            roi_feats = self.roi_align(cls_scores, rois) # [N_pos, C, 7, 7]
            f_vis = roi_feats.mean(dim=[2, 3]) # Mean Pooling -> [N_pos, C] (256)
            
            # 5.2 提取几何特征 F_geo [w, h, r, a]
            w = (pos_decoded_bboxes[:, 2] - pos_decoded_bboxes[:, 0]).unsqueeze(1)
            h = (pos_decoded_bboxes[:, 3] - pos_decoded_bboxes[:, 1]).unsqueeze(1)
            # 归一化 (假设图像 640x640，粗略归一化)
            img_h, img_w = batch_img_metas[0]['img_shape'][:2]
            norm_w, norm_h = w / img_w, h / img_h
            f_geo = torch.cat([norm_w, norm_h, norm_w/norm_h, norm_w*norm_h], dim=1)
            
            # 5.3 融合并过 MLP -> Query Embedding
            query_input = torch.cat([f_vis, f_geo], dim=1) # [N_pos, 260]
            e_query = self.proto_mlp(query_input) # [N_pos, 64]
            
            # 5.4 准备 Key Embedding (Prototypes)
            # 原型库：[Num_Classes, 256] + [Num_Classes, 4]
            proto_input = torch.cat([self.proto_vis, self.proto_geo], dim=1) # [C, 260]
            e_key = self.proto_mlp(proto_input) # [C, 64] (共享 MLP)
            
            # [cite_start]5.5 计算 Contrastive Loss [cite: 127-130]
            loss_con = self.loss_con(e_query, e_key, pos_gt_labels)
        else:
            loss_con = flatten_iou_preds.sum() * 0

        # -----------------------------------------------------------
        # [cite_start]Step 6: 计算 Topology Energy Loss [cite: 131-140]
        # -----------------------------------------------------------
        # 必须按图片分组计算 (Image-wise)
        loss_energy = 0.0
        
        # 拆分预测框和 GT 框回 Batch 维度
        current_idx = 0
        for i, num_pos in enumerate(num_pos_per_img):
            if num_pos < 2: # 少于2个目标无法构成图
                current_idx += num_pos
                continue
                
            # 获取当前图片的预测框和目标框
            # 切片: [current : current+num]
            img_pred_bboxes = pos_decoded_bboxes[current_idx : current_idx + num_pos]
            img_target_bboxes = pos_gt_bboxes[current_idx : current_idx + num_pos]
            
            # 计算当前图片的拓扑损失 (核心创新点)
            loss_energy += self.loss_energy(img_pred_bboxes, img_target_bboxes)
            
            current_idx += num_pos

        loss_energy = loss_energy / len(batch_img_metas) # Average over batch

        # -----------------------------------------------------------
        # Step 7: 汇总返回
        # -----------------------------------------------------------
        # 注意：这里需要补全 loss_cls 和 loss_bbox 的计算，
        # 建议直接复制 YOLOWorldHead.loss_by_feat 的相关代码。
        
        losses = dict(
            # loss_cls=loss_cls,   # 需补全
            # loss_bbox=loss_bbox, # 需补全
            loss_iou_branch=loss_iou_branch,
            loss_con=loss_con,
            loss_energy=loss_energy
        )
        return losses


@MODELS.register_module()
class YOLOHeadWithGraphLoss(YOLOWorldHead):
    
    def __init__(self,
                 prototype_cfg: ConfigDict,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        # 加载 Loss 和 原型
        self.prototypes = self._load_prior(prototype_cfg['path'])
        self.use_graph_loss = prototype_cfg.get('use_graph_loss', True)
        self.weight_graph_loss = prototype_cfg.get('weight_graph_loss', 1.0)


    def _load_prior(self, path):
        """
        加载原型库，适配用户保存的字典结构:
        Keys: ['class_ids', 'vis_prototypes', 'similarity_matrix','similarity_matrix_method_2']
        """
        # 加载 .pt 文件
        data = torch.load(path, map_location='cpu')
        
        # 1. 维度检查
        # visual prototypes shape: [N, 512]
        vis_dim = data['vis_prototypes'].shape[1]
        assert vis_dim == 512, f"Expected visual prototype dim 512, got {vis_dim}"
        
        # 2. 注册为 Buffer (随模型移动到 GPU，不更新参数)
        # 注意：这里我们修改了键名以匹配你的保存代码
        self.register_buffer('proto_vis', data['vis_prototypes'])  # [6, 512]
        self.register_buffer('proto_ids', data['class_ids'])       # [6]
        self.register_buffer('similarity_matrix', data['similarity_matrix']) # [6, 6]
        self.register_buffer('similarity_matrix_method_2', data['similarity_matrix_method_2']) # [6, 6]
        
        # 3. (可选) 打印加载信息
        print(f"[UltrasoundHead] Loaded prototypes: {len(data['class_ids'])} classes.")
        print(f" - Visual Shape: {self.proto_vis.shape}")
        print(f" - Similarity Matrix Shape: {self.similarity_matrix.shape}")
        
        return data

    def loss(self, img_feats, txt_feats, txt_masks, batch_data_samples):
        """
        重写 loss，确保参数传递顺序与 loss_by_feat 严格一致
        """
        # 1. 前向传播 (返回训练时 cls, bbox, dist         推理时：cls, bbox)
        outs = self(img_feats, txt_feats, txt_masks)
        
        # 2. 构造输入元组 (Strict Order)
        loss_inputs = (
            img_feats,                          # 1. 特征图
        ) + outs + (                            # 2,3,4,5. 网络输出 (cls, bbox, dist, iou)
            batch_data_samples['bboxes_labels'],# 6. GT 实例
            batch_data_samples['img_metas'],    # 7. 图片元信息
            txt_masks                           # 8. [修复] 补上文本掩码
        )
        
        # 3. 计算损失
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self, 
                     img_feats,
                     cls_scores, 
                     bbox_preds, 
                     bbox_dist_preds,
                     batch_gt_instances, 
                     batch_img_metas,
                     batch_text_masks=None,
                     batch_gt_instances_ignore=None):
       
        num_imgs = len(batch_img_metas)
        device = cls_scores[0].device

        # ==================================================================
        # PART 1: 官方 YOLO-World 预处理逻辑 (原样复刻)
        # ==================================================================
        
        # 1.1 动态 Anchor 生成
        current_featmap_sizes = [c.shape[2:] for c in cls_scores]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train, dtype=cls_scores[0].dtype,
                device=device, with_stride=True)
            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # 1.2 GT 预处理
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # 1.3 Flatten 预测值
        flatten_cls_preds = [
            c.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for c in cls_scores
        ]
        flatten_pred_bboxes = [
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for b in bbox_preds
        ]

        flatten_pred_dists = [
            d.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for d in bbox_dist_preds
        ]
        
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)

        # 1.4 解码 (Decode)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        # 1.5 正负样本分配 (Assigner)
        assigned_result = self.assigner(
            (flatten_decoded_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), 
            self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # ==================================================================
        # PART 2: 官方 Loss 计算 (Cls + Bbox + DFL)
        # ==================================================================
        
        # 2.1 Classification Loss (含 Text Mask 逻辑)
        # print(flatten_cls_preds.shape)
        if batch_text_masks is not None:
            cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
                -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            loss_cls = (loss_cls * cls_weight).sum()
        else:
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # **重要提示**：在归一化之前，备份一份绝对坐标用于 Topology Loss
        # 因为 loss_bbox 计算通常会执行 /= stride，破坏绝对坐标
        decoded_bboxes_for_topology = flatten_decoded_bboxes.clone() 

        # 2.2 Bbox Loss & DFL Loss
        # 归一化预测框
        assigned_bboxes /= self.stride_tensor
        flatten_decoded_bboxes /= self.stride_tensor

        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_decoded_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            
            # IoU Loss (CIoU)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # DFL Loss (官方逻辑)
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1, eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0

        # ==================================================================
        # PART 3: 你的 Loss 计算
        # ==================================================================
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        if not self.use_graph_loss:
            return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

        roi_extractor = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[16, 32]
        ).to(device)
        
        prior_sim = self.prototypes['similarity_matrix_method_2'].to(device)#similarity_matrix_method_2
        # prior_sim = self.prototypes['similarity_matrix'].to(device)#similarity_matrix_method
        # 3.2 准备 Metric 和 Topology 数据
        fg_mask_per_img = fg_mask_pre_prior.view(num_imgs, -1)
        num_pos_per_img = fg_mask_per_img.sum(dim=1).int().tolist()

        # 构建 Batch Index 用于 RoI Align
        batch_inds_list = []
        for i, n in enumerate(num_pos_per_img):
            batch_inds_list.append(torch.full((n, 1), i, device=device))
        
        if num_pos > 0:
            batch_inds = torch.cat(batch_inds_list, dim=0)
            # 使用备份的绝对坐标
            pos_abs_pred = decoded_bboxes_for_topology[fg_mask_pre_prior]
            rois = torch.cat([batch_inds, pos_abs_pred], dim=1) # [N, 5]

            # --- Metric Branch (Contrastive) ---
            # A. 视觉特征 (512维)
            """
            Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
            """
            target_feat = img_feats[1:] 
            roi_feats = roi_extractor(target_feat, rois)
            vis_vecs = F.adaptive_avg_pool2d(roi_feats, (1, 1)).flatten(1)
            #############################提取了所有框的特征###########################################
            # 遍历每一张图片，提取每一张图片的每一个类别的score为TopK的特征
            labels_list = assigned_result['assigned_labels'][fg_mask_pre_prior]
            loss_graph = flatten_cls_preds.new_zeros(1)
            for i, n in enumerate(num_pos_per_img):
                if n < 1:
                    continue
                img_id = i
                start = sum(num_pos_per_img[:i])
                end = sum(num_pos_per_img[:i+1])
                max_values_scores, max_indices = torch.max(assigned_scores[i][fg_mask_pre_prior[i]], dim=1)
                img_feat = vis_vecs[start:end]
                cls_list = labels_list[start:end]
                # print(cls_list)
                # print(max_indices)
                cls_topk_feats={}
                cls_topk_indices={}
                res_indices = self._get_top_k_indices_per_class(cls_list,max_values_scores)
                #按照cls_id排序
                res_indices = dict(sorted(res_indices.items()))
                for cls_id, indices in res_indices.items():
                    # print(f"Image {i}, Class {cls_id}, Top-K Indices: {indices}")
                    img_feat_topk = img_feat[indices]#类别cls_id的topk特征
                    # print("===================================================")
                    cls_topk_feats[cls_id]=img_feat_topk
                    cls_topk_indices[cls_id]=indices

                all_loss, _  = self._compute_combinations_loss(cls_topk_feats, prior_sim)
                min_values, _ = torch.topk(all_loss, k=min(3,len(all_loss)), largest=False)
                total_sum = min_values.sum()
                loss_graph += total_sum
                # print(loss_graph)
                # loss_graph += all_loss.mean()
        else:
            loss_graph = flatten_cls_preds.new_zeros(1)
        
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
            loss_graph=loss_graph * world_size * 10
        )

    def _get_top_k_indices_per_class(self, labels, scores, k=3):
        """
        labels: [n] 类别张量
        scores: [n] 置信度张量
        k: 每个类别取的数量
        """
        # 1. 获取所有存在的类别
        unique_classes = labels.unique()
        counts = []
        for cls in unique_classes:
            count = (labels == cls).sum().item()
            counts.append(count)
        
        # 3. 确定最终统一的 k 值 (final_k)
        # 取：期望值k 与 所有类别中最小样本数 之间的最小值
        min_available = min(counts)
        final_k = min(k, min_available)
        # 用于存储结果：{类别ID: 全局索引张量}
        res_indices = {}

        for cls in unique_classes:
            # 2. 找到属于当前类别的所有全局索引
            # torch.where 返回的是一个元组，取第一个元素
            cls_indices = torch.where(labels == cls)[0]
            
            # 3. 提取这些样本的分数
            cls_scores = scores[cls_indices]
            
            # 4. 确定实际能取的 k 值（防止样本数小于 k）
            # actual_k = min(k, cls_scores.size(0))
            
            # 5. 在该类别的分数中找 top-k
            # topk_local_indices 是在 cls_scores 里的索引 (0, 1, 2...)
            _, topk_local_indices = torch.topk(cls_scores, k=final_k)
            
            # 6. 映射回原始的全局索引
            topk_global_indices = cls_indices[topk_local_indices]
            
            res_indices[cls.item()] = topk_global_indices

        return res_indices

    def _compute_combinations_loss(self, feature_dict, prior_sim):
        """
        feature_dict: 字典，key为类别，value为 [3, 512] 的 tensor
        prior_sim: 预先写好的 [n, n] 矩阵
        """
        # 1. 准备数据：将字典转换为形状为 [n, 3, 512] 的 Tensor
        categories = list(feature_dict.keys())
        n = len(categories)
        # [n, 3, 512]
        all_features = torch.stack([feature_dict[c] for c in categories])
        
        k = all_features.shape[1]
        # 为了高效查找，可以将全局ID列表转换为list
        all_proto_ids_list = self.proto_ids.cpu().tolist()
    
        try:
            # 查找当前类别在全局ID列表中的索引
            indices_in_prior = [all_proto_ids_list.index(cat_id) for cat_id in categories]
        except ValueError as e:
            # 异常处理：如果图片中的某个类别ID不在全局原型中，这通常不应该发生
            print(f"错误：图片中的类别ID {e} 未在全局原型ID列表中找到。")
            return torch.tensor(0.0, device=prior_sim.device), None

        # 使用高级索引从 prior_sim 中提取子矩阵
        # 首先按行索引
        sub_prior_sim_rows = prior_sim[indices_in_prior]
        # 然后按列索引，得到一个 [n, n] 的目标矩阵
        target_sim_matrix = sub_prior_sim_rows[:, indices_in_prior]


        # 2. 生成所有组合的索引
        # 比如 n=2, 则生成 [[0,0], [0,1], [0,2], [1,0], [1,1] ... [2,2]]
        # 形状为 [3^n, n]
        coords = [torch.arange(k) for _ in range(n)]
        # indices = torch.cartesian_product(*coords) 
        grid = torch.meshgrid(*coords, indexing='ij') 
        indices = torch.stack(grid, dim=-1).reshape(-1, len(coords))
        num_combinations = indices.shape[0]
        
        # print(f"检测到类别数 n={n}, 总组合数 3^n = {num_combinations}")

        # 3. 提取特征并构建 Batch
        # 我们要利用 indices 从 all_features 中提取特征
        # 这里的技巧是：将 all_features 展平，或者使用索引映射
        # 生成的 batch_features 形状为 [num_combinations, n, 512]
        
        # 创建一个辅助索引，对应每个类别在 all_features 的第一维
        category_range = torch.arange(n).repeat(num_combinations, 1) # [num_combinations, n]
        
        # 提取特征：从 [n, 3, 512] 中选出对应的组合
        # 结果形状: [num_combinations, n, 512]
        batch_features = all_features[category_range, indices]
        
        # 4. 计算相似度矩阵
        # 先对特征进行归一化，这样点积就是余弦相似度
        batch_features = F.normalize(batch_features, p=2, dim=2)
        
        # 使用批量矩阵乘法 bmm 计算所有组合的 [n, n] 矩阵
        # [B, n, 512] * [B, 512, n] -> [B, n, n]
        sim_matrices = torch.bmm(batch_features, batch_features.transpose(1, 2))
        
        # 5. 计算损失
        # 将 prior_sim 扩展到同样的 batch size 方便计算
        # prior_sim 形状: [n, n] -> [num_combinations, n, n]
        # target_sim = prior_sim.unsqueeze(0).expand(num_combinations, -1, -1)
        target_sim = target_sim_matrix.unsqueeze(0).expand(num_combinations, -1, -1)
        
        # 计算均方误差 (MSE) 或其他损失，假设在矩阵维度上计算
        # 结果是一个长度为 3^n 的 tensor，表示每种组合的损失
        all_losses = F.mse_loss(sim_matrices, target_sim, reduction='none').mean(dim=(1, 2))
        return all_losses, indices