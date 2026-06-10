# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
import os
import sys
import cv2
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmyolo.utils import register_all_modules
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

def inference(model, image, texts, test_pipeline, score_thr=0.001 , max_dets=100):
    image = cv2.imread(image)
    image = image[:, :, [2, 1, 0]]
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores


if __name__ == "__main__":

    config_file = "/media/Storage3/wmm/ICML/Medical-OD/configs/stage1_finetune_3VT_10_shot.py"
    checkpoint = "/media/Storage3/wmm/ICML/Medical-OD/work_dirs/stage1_finetune_3VT_10_shot/20260107_121109_train/best_coco_bbox_mAP_50_epoch_480.pth"

    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    # init model
    cfg.load_from = checkpoint
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    texts =[["Ascending Aorta"],
            ["Spine"],
            ["Pulmonary trunk & ductus arteriosus"],
            ["Trachea"],
            ["Superior vena cava"],
            ["Arch of Aorta"]]

    image = "/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal/1.2.410.200001.1.1131.3729117517.3.20220325.1161732904.186.54.jpg"
    print(f"starting to detect: {image}")
    results = inference(model, image, texts, test_pipeline)
    format_str = [
        f"obj-{idx}: {box}, label-{lbl}, class-{lbl_text}, score-{score}"
        for idx, (box, lbl, lbl_text, score) in enumerate(zip(*results))
    ]
    print("detecting results:")
    for q in format_str:
        print(q)

    class_name = ['Ascending Aorta','Spine','Pulmonary trunk & ductus arteriosus','Trachea','Superior vena cava','Arch of Aorta']
    # visualize
    img = cv2.imread(image)
    boxes, lbl, label_texts, scores = results
    lbl = [int(x) for x in lbl]
    label_name = [class_name[int(x)] for x in lbl]
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0), (0, 128, 128),
        (128, 0, 128), (128, 128, 0)
    ]
    for box, score, label, int_label in zip(boxes, scores, label_name, lbl):
        box = box.astype(int)
        x1, y1, x2, y2 = box
        color = colors[int_label % len(colors)]
        # draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # draw label
        label_text = f'{label}: {score:.2f}'
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - label_height - baseline),
                      (x1 + label_width, y1), color, -1)
        cv2.putText(img,
                    label_text, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    # save image
    if not osp.exists('demo/vis'):
        os.makedirs('demo/vis')
    out_file = osp.join('demo/vis', osp.basename(image))
    cv2.imwrite(out_file, img)
    print(f"visual results are saved at {out_file}")

