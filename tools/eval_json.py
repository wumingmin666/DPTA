import argparse
import os
import json
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# python ./tools/analysis_tools/coco_error_analysis.py /media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage1_finetune_3VT_10_shot/20260107_115344_text_iou0.4_score0.001/test_results.bbox.json --ann /media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/test/annotation_name.json
# python ./tools/eval_json.py /media/Storage3/wmm/ICML/Medical-OD/configs/stage2_finetune_spine_10_shot.py  /media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage2_finetune_spine_10_shot/20260114_202149_iou0.4_score0.001/test_results.bbox.json --ann-file /media/Storage3/wmm/ICML/data/spine/GE/test.json
# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate mAP and Recall from JSON result and Config')
#     parser.add_argument('config', default="/media/Storage3/wmm/ICML/YOLO-World-Plus/configs/configs_few_shot/finetune_3VT_10_shot_plus.py" ,help='OpenMMLab config file path (.py)')
#     parser.add_argument('result', default= "/media/Storage3/wmm/ICML/YOLO-World-Plus/work_dirs2/finetune_3VT_10_shot/20251216_203517_未矫正/test_results.bbox.json", help='Prediction result json file path (.json)')
#     parser.add_argument(
#         '--ann-file', 
#         type=str, 
#         default=None, 
#         help='Optional: Force specify annotation file path if script fails to find it in config'
#     )
#     return parser.parse_args()

# def get_ann_file_from_config(config_path):
#     """
#     尝试从 OpenMMLab 配置文件中解析 test/val 的标注文件路径。
#     兼容 MMDetection V2.x (mmcv) and V3.x (mmengine)。
#     """
#     try:
#         # 尝试使用 mmengine (MMDetection 3.x)
#         from mmengine.config import Config
#         cfg = Config.fromfile(config_path)
#         version = 3
#     except ImportError:
#         # 回退到 mmcv (MMDetection 2.x)
#         try:
#             from mmcv import Config
#             cfg = Config.fromfile(config_path)
#             version = 2
#         except ImportError:
#             print("Error: Could not import mmengine or mmcv. Please install openmmlab dependencies.")
#             sys.exit(1)

#     ann_file = None
    
#     # === 解析逻辑 ===
#     if version == 3:
#         # MMDetection 3.x 结构通常在 test_dataloader -> dataset -> ann_file
#         if hasattr(cfg, 'test_dataloader'):
#             dataset = cfg.test_dataloader.dataset
#             if 'ann_file' in dataset:
#                 ann_file = dataset.ann_file
#         # 如果没有 test_dataloader，尝试 val_dataloader
#         if ann_file is None and hasattr(cfg, 'val_dataloader'):
#             dataset = cfg.val_dataloader.dataset
#             if 'ann_file' in dataset:
#                 ann_file = dataset.ann_file
                
#     elif version == 2:
#         # MMDetection 2.x 结构通常在 data -> test -> ann_file
#         if hasattr(cfg, 'data'):
#             if 'test' in cfg.data and 'ann_file' in cfg.data.test:
#                 ann_file = cfg.data.test.ann_file
#             elif 'val' in cfg.data and 'ann_file' in cfg.data.val:
#                 ann_file = cfg.data.val.ann_file

#     # 处理数据根目录 data_root (如果存在)
#     data_root = getattr(cfg, 'data_root', None)
#     if ann_file and data_root and not os.path.isabs(ann_file):
#         ann_file = os.path.join(data_root, ann_file)

#     return ann_file

# def main():
#     args = parse_args()

#     # 1. 获取 Ground Truth (标注文件路径)
#     if args.ann_file:
#         ann_file = args.ann_file
#     else:
#         print(f"Loading config from: {args.config} ...")
#         ann_file = get_ann_file_from_config(args.config)
    
#     if not ann_file:
#         print("Error: Could not find 'ann_file' path in config. Please specify it manually using --ann-file.")
#         sys.exit(1)
        
#     print(f"Ground Truth Annotation file: {ann_file}")
#     print(f"Prediction Result file: {args.result}")

#     if not os.path.exists(ann_file):
#         print(f"Error: Annotation file not found at {ann_file}")
#         sys.exit(1)
#     if not os.path.exists(args.result):
#         print(f"Error: Result file not found at {args.result}")
#         sys.exit(1)

#     # 2. 初始化 COCO 对象
#     try:
#         cocoGt = COCO(ann_file)        # 加载标注
#         cocoDt = cocoGt.loadRes(args.result) # 加载预测结果
#     except Exception as e:
#         print(f"Error loading COCO data: {e}")
#         print("Ensure your JSON result format matches standard COCO result format.")
#         sys.exit(1)

#     # 3. 运行评估
#     # iouType 默认为 'bbox'，如果是实例分割可以使用 'segm'
#     cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    
#     print("\nRunning COCO Evaluation...")
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()

#     # 4. 解析并打印关键指标 (可选，summarize 已经打印了)
#     stats = cocoEval.stats
#     # stats[0] = mAP (IoU=0.50:0.95)
#     # stats[1] = mAP (IoU=0.50)
#     # stats[8] = AR (Recall) @ maxDets=100
#     print("\n" + "="*30)
#     print("Summary Extraction:")
#     print(f"mAP (0.5:0.95): {stats[0]:.4f}")
#     print(f"mAP (0.5)     : {stats[1]:.4f}")
#     print(f"Recall (AR@100): {stats[8]:.4f}")
#     print("="*30)

# if __name__ == '__main__':
#     main()

import argparse
import os
import sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib

# 用于屏蔽 summarize() 的打印输出，以免产生误导性文字
@contextlib.contextmanager
def SuppressStdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate mAP and Recall (specific IoU) from JSON')
    parser.add_argument('config', help='OpenMMLab config file path (.py)')
    parser.add_argument('result', help='Prediction result json file path (.json)')
    parser.add_argument('--ann-file', type=str, default=None, help='Force specify annotation file path')
    return parser.parse_args()

def get_ann_file_from_config(config_path):
    # (保持原有的解析逻辑不变)
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        version = 3
    except ImportError:
        try:
            from mmcv import Config
            cfg = Config.fromfile(config_path)
            version = 2
        except ImportError:
            print("Error: Please install openmmlab dependencies (mmcv/mmengine).")
            sys.exit(1)

    ann_file = None
    if version == 3:
        if hasattr(cfg, 'test_dataloader'):
            dataset = cfg.test_dataloader.dataset
            if 'ann_file' in dataset: ann_file = dataset.ann_file
        if ann_file is None and hasattr(cfg, 'val_dataloader'):
            dataset = cfg.val_dataloader.dataset
            if 'ann_file' in dataset: ann_file = dataset.ann_file     
    elif version == 2:
        if hasattr(cfg, 'data'):
            if 'test' in cfg.data and 'ann_file' in cfg.data.test:
                ann_file = cfg.data.test.ann_file
            elif 'val' in cfg.data and 'ann_file' in cfg.data.val:
                ann_file = cfg.data.val.ann_file

    data_root = getattr(cfg, 'data_root', None)
    if ann_file and data_root and not os.path.isabs(ann_file):
        ann_file = os.path.join(data_root, ann_file)
    return ann_file

def main():
    args = parse_args()

    # 1. 寻找标注文件
    if args.ann_file:
        ann_file = args.ann_file
    else:
        ann_file = get_ann_file_from_config(args.config)
    
    if not ann_file or not os.path.exists(ann_file):
        print(f"Error: Annotation file not found: {ann_file}")
        sys.exit(1)
        
    print(f"Ground Truth: {ann_file}")
    print(f"Result JSON : {args.result}")

    # 2. 加载数据
    print("Loading COCO data...")
    cocoGt = COCO(ann_file)
    try:
        cocoDt = cocoGt.loadRes(args.result)
    except Exception as e:
        print(f"Error loading result JSON: {e}")
        sys.exit(1)

    # =======================================================
    # 第一步：标准评估 (为了得到 mAP 0.5:0.95 和 mAP 0.5)
    # =======================================================
    print("\n[1] Running Standard Evaluation (IoU=0.50:0.95)...")
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize() # 标准输出，显示详细信息
    
    standard_stats = cocoEval.stats[:] # 复制一份，防止被覆盖

    # =======================================================
    # 第二步：特定 IoU=0.5 的评估 (为了得到 Recall@0.5)
    # =======================================================
    print("\n[2] Running Specific Evaluation (IoU=0.50 Only)...")
    cocoEval_iou05 = COCOeval(cocoGt, cocoDt, iouType='bbox')
    
    # --- 核心 Hack：只设定 IoU=0.5 ---
    cocoEval_iou05.params.iouThrs = np.array([0.5])
    
    cocoEval_iou05.evaluate()
    cocoEval_iou05.accumulate()
    
    # --- 关键修正：必须调用 summarize() 才能生成 stats，但我们屏蔽它的输出 ---
    with SuppressStdout():
        cocoEval_iou05.summarize()
    
    stats_05 = cocoEval_iou05.stats
    
    # 此时 stats_05[8] 就是 AR @ maxDets=100 @ IoU=0.5
    recall_iou05 = stats_05[8]

    # =======================================================
    # 第三步：汇总输出
    # =======================================================
    print("\n" + "="*40)
    print("FINAL METRICS SUMMARY")
    print("="*40)
    # standard_stats[0] 是 AP @ IoU=0.50:0.95
    # standard_stats[1] 是 AP @ IoU=0.50
    print(f"mAP (IoU=0.50:0.95) : {standard_stats[0]:.4f}")
    print(f"mAP (IoU=0.50)      : {standard_stats[1]:.4f}")
    print("-" * 40)
    # standard_stats[8] 是 AR @ IoU=0.50:0.95 (默认)
    print(f"Recall (IoU=0.50:0.95): {standard_stats[8]:.4f} (Standard AR)")
    print(f"Recall (IoU=0.50)     : {recall_iou05:.4f} <--- 你需要的指标")
    print("="*40)

if __name__ == '__main__':
    main()