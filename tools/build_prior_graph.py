import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.dataset import Compose, default_collate

from mmyolo.registry import MODELS, DATASETS
from mmyolo.utils import register_all_modules

from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

# 获取 build_prototypes.py 所在的绝对路径
file_path = os.path.abspath(__file__)
# 获取 tools 目录路径
tools_dir = os.path.dirname(file_path)
# 获取项目根目录 (tools 的上一级)
project_root = os.path.dirname(tools_dir)

# 将项目根目录插入到 sys.path 的最前面，确保优先加载
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import yolo_world
# import med_yolo
register_all_modules(init_default_scope=True)


config = "/media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_3VT_20_shot_text_k9.py"
# checkpoint = "/media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage1_finetune_3VT_10_shot/20260107_121109_train/best_coco_bbox_mAP_50_epoch_480.pth"
# checkpoint = '/media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage1_finetune_3VT_10_shot/20260107_143403_train_text1/best_coco_bbox_mAP_50_epoch_900.pth'
checkpoint = '/media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage1_finetune_3VT_20_shot_text_k9/best_coco_bbox_mAP_50_epoch_360.pth'
# config = "/media/Storage3/wmm/ICML/Medical-OD/configs/stage_finetune_spine_10_shot.py"
out = "/media/Storage3/wmm/ICML/Medical-OD/checkpoints_text/demo1_k9_20shot_prior.pt"
# checkpoint="/media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage1_finetune_spine_10_shot/20260112_132313_train/best_coco_bbox_mAP_50_epoch_400.pth"
device = "cuda:1"

def main():
    cfg = Config.fromfile(config)

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
     
    model = MODELS.build(cfg.model)
    
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()

    # cls_preds = model.bbox_head.head_module.cls_preds

    # RoI Extractor 属于 mmdet 组件，但我们可以直接用类实例化，
    # 或者用 MODELS.build (mmyolo 可以访问 mmdet 组件)
    roi_extractor = SingleRoIExtractor(
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=512,
        featmap_strides=[16, 32]
    ).to(device)

    
    vis_features_bank = defaultdict(list) #{label_id: [vis_feature1, vis_feature2, ...], ...}# 用于计算原型，之后对原型计算相似性构建边
    all_images_data = []# 先对每张图片构建graph，之后对边求均值
    


    # pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    pipeline = Compose(cfg.train_dataloader.dataset.pipeline)
    #没有构建dataloader，直接用dataset遍历
    progress_bar = tqdm(total=len(dataset))
    with torch.no_grad():
        for idx in range(len(dataset)):
            data_info = dataset.get_data_info(idx)
            data_batch = pipeline(data_info)#导致texts的顺序发生变化
            # [关键修改] 使用 model.data_preprocessor 进行标准化和类型转换
            # 1. 构造符合 preprocessor 输入要求的字典 (inputs 放入列表)
            data = {
                'inputs': [data_batch['inputs']], #inputs的shape为[3,64,64]
                'data_samples': [data_batch['data_samples']] #元信息
            }
            
            # 2. 调用 preprocessor (自动处理 .to(device), .float(), 归一化等)
            data = model.data_preprocessor(data, training=False)#Perform normalization, padding and bgr2rgb conversion

            inputs = data['inputs']
            data_samples = data['data_samples']
            results = model.extract_feat(inputs, data_samples)#经过neck后的结果
            
            # 解包：我们只需要第一个元素 img_feats (视觉特征金字塔)
            img_feats, txt_feats, txt_masks = results
        
            '''
            img_feats: list of 3 tensors, each is [1, C, H, W]
            img_feats[0]: P3, stride=8 [1,256,80,80]
            img_feats[1]: P4, stride=16 [1,512,40,40]
            img_feats[2]: P5, stride=32 [1,512,20,20]
            txt_feats: [1, num_classes, 512]
            '''
            # 要使用YOLOWorldHeadModule中的模块self.cls_pred获得最终的img_feats,通道都是512
            
            # feats_for_roi = tuple(module(img_feats[i]) for i, module in enumerate(cls_preds))
            # YOLO World Neck 输出通常是 [P3(128), P4(256), P5(256)]
            # RoI Extractor 需要统一通道，所以我们切片取后两层 [P4, P5]
            # 对应的 Strides 必须是 [16, 32]
            feats_for_roi = img_feats[1:]
            # -----------------------------------------------------------------

            gt_instances = data_samples[0].gt_instances
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels
            
            if len(gt_bboxes) == 0:
                progress_bar.update(1)
                continue

            batch_inds = torch.zeros((gt_bboxes.size(0), 1), device=device)
            rois = torch.cat([batch_inds, gt_bboxes], dim=1)

            roi_feats = roi_extractor(feats_for_roi, rois)# [N,512,7,7],会根据ROI的大小选择取哪一个特征图取提取
            vis_vecs = F.adaptive_avg_pool2d(roi_feats, (1, 1)).flatten(1)#[N,512,7,7]-->[N,512,1,1]-->[N,512]
            #==============================ROI视觉特征和文本特征的相似度==============================================
            # print("==========================================")
            # print("img_id: ", data_samples[0].img_id)
            # print("gt_labels: ",gt_labels)
            # print("texts: ", data_samples[0].texts)
            # txt_feats_squeezed = txt_feats.squeeze(0)
            # vis_vecs_norm = F.normalize(vis_vecs, p=2, dim=1)
            # txt_feats_norm = F.normalize(txt_feats_squeezed, p=2, dim=1)
            # similarity_matrix_vis_txt = torch.matmul(vis_vecs_norm, txt_feats_norm.T)
            # print(similarity_matrix_vis_txt)
            #============================================================================


            for i, label in enumerate(gt_labels):
                label_id = label.item()
                vis_features_bank[label_id].append(vis_vecs[i].cpu())
            
            # 方法二使用：先对每张图片构建graph，之后对边求均值
            current_image_features = []
            for i, label in enumerate(gt_labels):
                label_id = label.item()
                # 将特征保存在 CPU 上以节省显存
                current_image_features.append((label_id, vis_vecs[i].cpu()))

            # 如果图片中有标注，则进行排序并保存
            if current_image_features:
                # 关键步骤: 根据 label_id (元组的第一个元素) 对列表进行排序
                current_image_features.sort(key=lambda x: x[0])

                # 将排序后的元组列表解包成两个独立的 tensor
                sorted_labels = torch.tensor([item[0] for item in current_image_features], dtype=torch.long)
                sorted_features = torch.stack([item[1] for item in current_image_features])

                # 将这张图片的已排序数据存入总列表
                all_images_data.append({
                    'image_idx': idx,
                    'labels': sorted_labels,
                    'vis_features': sorted_features
                })

            progress_bar.update(1)

    print("\nAggregating prototypes...")
    final_prototypes = {
        'class_ids': [],
        'vis_prototypes': []
    }
    
    all_class_ids = sorted(vis_features_bank.keys())
    for cls_id in all_class_ids:
        vis_tensor = torch.stack(vis_features_bank[cls_id])
        
        vis_proto = vis_tensor.mean(dim=0)
        
        final_prototypes['class_ids'].append(cls_id)
        final_prototypes['vis_prototypes'].append(vis_proto)

    final_prototypes['class_ids'] = torch.tensor(final_prototypes['class_ids'])
    final_prototypes['vis_prototypes'] = torch.stack(final_prototypes['vis_prototypes'])
    
    #利用原型构建图，因为前面有对class_id排序，所有这里的相似度矩阵行列顺序和class_ids是一致的
    vis_prototypes = final_prototypes['vis_prototypes']
    # 步骤 1: 对所有原型向量进行 L2 归一化
    vis_prototypes_norm = F.normalize(vis_prototypes, p=2, dim=1)
    # 步骤 2: 计算归一化后的矩阵与其自身的转置相乘
    similarity_matrix = torch.matmul(vis_prototypes_norm, vis_prototypes_norm.T)
    # 将计算出的相似度矩阵添加到要保存的字典中
    final_prototypes['similarity_matrix'] = similarity_matrix
    print(f"Similarity matrix calculated with shape: {similarity_matrix.shape}")
    print(f"Similarity matrix preview:\n{similarity_matrix}")


    #方法二：利用每张图片的特征构建图，然后对边取均值
    edge_sums = defaultdict(float)
    edge_counts = defaultdict(int)
    for image_data in all_images_data:
        labels = image_data['labels']
        vis_features = image_data['vis_features']
        num_instances = labels.size(0)

        # 对当前图片的每一对实例计算相似度
        for i in range(num_instances):
            for j in range(num_instances):
                # if i != j:
                label_i = labels[i].item()
                label_j = labels[j].item()
                feat_i = F.normalize(vis_features[i].unsqueeze(0), p=2, dim=1)
                feat_j = F.normalize(vis_features[j].unsqueeze(0), p=2, dim=1)
                sim_ij = torch.matmul(feat_i, feat_j.T).item()
                if i==1 and j==5:
                    print(sim_ij)
                edge_sums[(label_i, label_j)] += sim_ij
                edge_counts[(label_i, label_j)] += 1
                # print(f"Image {image_data['image_idx']} - Edge ({label_i}, {label_j}): sim={sim_ij:.4f}")
                # if image_data['image_idx'] ==0 and i == 0 and j == 0:  # 仅打印每张图片的第一个实例对以减少输出量
                #     print(f"  Feature {i}: {vis_features[i]}")
                
    # 计算最终的相似度矩阵
    similarity_matrix_2 = torch.zeros((len(all_class_ids), len(all_class_ids)))
    class_id_to_index = {cls_id: idx for idx, cls_id in enumerate(all_class_ids)}
    for (label_i, label_j), sim_sum in edge_sums.items():
        count = edge_counts[(label_i, label_j)]
        avg_sim = sim_sum / count
        idx_i = class_id_to_index[label_i]
        idx_j = class_id_to_index[label_j]
        similarity_matrix_2[idx_i, idx_j] = avg_sim
    #查看结果
    print(f"\nSimilarity matrix (method 2) calculated with shape: {similarity_matrix_2.shape}")
    print(f"Similarity matrix (method 2) preview:\n{similarity_matrix_2}")  
    final_prototypes['similarity_matrix_method_2'] = similarity_matrix_2

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(final_prototypes, out)
    print(f"\nPrototype library saved to {out}")

if __name__ == '__main__':
    main()

# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python tools/build_prototypes.py /media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_3VT_20_shot.py /media/Storage3/wmm/ICML/Medical-OD/checkpoints/best_coco_bbox_mAP_50_epoch_380.pth