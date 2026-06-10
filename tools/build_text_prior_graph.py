import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# -------------------------------------------------------------------------
# 动态获取项目根目录并加入系统路径
# -------------------------------------------------------------------------
file_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(file_path)
project_root = os.path.dirname(tools_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root added to path: {project_root}")

# -------------------------------------------------------------------------
# 导入必要的库
# -------------------------------------------------------------------------
try:
    import yolo_world
    print("Success: Local 'yolo_world' package imported.")
except ImportError as e:
    print(f"Error: Failed to import 'yolo_world'. Current sys.path: {sys.path}")
    raise e

from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

# -------------------------------------------------------------------------
# 配置参数（可修改）
# -------------------------------------------------------------------------
config = "/media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_EPFetal_5_shot.py"
out = "/media/Storage3/wmm/ICML/Medical-OD/checkpoints_text/EPFetal_5shot_text_prior.pt"
device = "cuda:1"


def main():
    print("=" * 60)
    print("Build Text Prior Graph: Computing Text Feature Similarity Matrix")
    print("=" * 60)
    
    # 注册所有模块
    register_all_modules(init_default_scope=True)
    
    # 加载配置文件
    print(f"\nLoading config from {config}...")
    cfg = Config.fromfile(config)
    
    # 提取类别信息
    classes = cfg.classes
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")
    
    # 提取文本编码器路径
    text_model_path = cfg.text_model
    print(f"Text model path: {text_model_path}")
    
    # -------------------------------------------------------------------------
    # 构建文本编码器
    # -------------------------------------------------------------------------
    print("\nBuilding text encoder...")
    
    from transformers import CLIPTextModel, CLIPTokenizer
    
    clip_text_encoder = CLIPTextModel.from_pretrained(text_model_path).to(device)
    clip_text_encoder.eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained(text_model_path)
    
    # -------------------------------------------------------------------------
    # 提取文本特征
    # -------------------------------------------------------------------------
    print("\nExtracting text features...")
    
    text_features_list = []
    
    with torch.no_grad():
        for cls_name in classes:
            # Tokenize
            text_inputs = clip_tokenizer(
                cls_name, 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors='pt'
            ).to(device)
            
            # Get text embeddings
            text_outputs = clip_text_encoder(**text_inputs)
            
            # 使用 pooler_output 或 last_hidden_state 的 [EOS] 位置
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_feat = text_outputs.pooler_output
            else:
                last_hidden = text_outputs.last_hidden_state
                attention_mask = text_inputs['attention_mask']
                eos_indices = attention_mask.sum(dim=1) - 1
                text_feat = last_hidden[torch.arange(len(eos_indices)), eos_indices]
            
            # 归一化（CLIP 的标准做法）
            text_feat = F.normalize(text_feat, p=2, dim=1)
            text_features_list.append(text_feat.cpu())
            
            print(f"  {cls_name}: {text_feat.shape}")
    
    # 堆叠所有文本特征
    txt_feats = torch.cat(text_features_list, dim=0)  # [N, D]
    print(f"\nText features shape: {txt_feats.shape}")
    
    # -------------------------------------------------------------------------
    # 构建类别 ID 映射
    # -------------------------------------------------------------------------
    class_ids = torch.arange(num_classes, dtype=torch.long)
    
    # -------------------------------------------------------------------------
    # 计算相似度矩阵
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing Similarity Matrices")
    print("=" * 60)
    
    # 方法一：原型级相似度
    print("\n[Method 1] Computing prototype-level similarity...")
    txt_feats_norm = F.normalize(txt_feats, p=2, dim=1)
    similarity_matrix = torch.matmul(txt_feats_norm, txt_feats_norm.T)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix preview:\n{similarity_matrix}")
    
    # 方法二：对于文本特征，退化为方法一
    print("\n[Method 2] Computing instance-level similarity...")
    print("Note: For text features, Method 2 is identical to Method 1.")
    similarity_matrix_method_2 = similarity_matrix.clone()
    
    # -------------------------------------------------------------------------
    # 保存结果（与 build_prior_graph.py 兼容的格式）
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    final_prototypes = {
        'class_ids': class_ids,                      # [N]
        'vis_prototypes': txt_feats,                 # [N, D] 用文本特征填充
        'similarity_matrix': similarity_matrix,      # [N, N]
        'similarity_matrix_method_2': similarity_matrix_method_2  # [N, N]
    }
    
    # 额外保存文本特征（可选）
    final_prototypes['txt_prototypes'] = txt_feats
    
    # 创建输出目录
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    # 保存
    torch.save(final_prototypes, out)
    print(f"\nText prior graph saved to: {out}")
    
    # 验证保存的文件
    print("\nVerifying saved file...")
    loaded = torch.load(out, map_location='cpu')
    print(f"Loaded keys: {list(loaded.keys())}")
    for key, value in loaded.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
