_base_ = (
    '../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
default_scope = 'mmyolo'
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)



num_classes = 6
num_training_classes = 6
max_epochs = 1000  # Maximum training epochs
val_interval_epochs=20
save_epoch_intervals = 20
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

load_from='/media/Storage3/wmm/ICML/Medical-OD/checkpoints/best_coco_bbox_mAP_50_epoch_380.pth'
persistent_workers = False
text_model = '/media/Storage3/wmm/ICML/YOLO-World-master/pretrained_models/clip-vit-base-patch32'

classes=('Ascending Aorta','Spine','Pulmonary trunk & ductus arteriosus','Trachea','Superior vena cava','Arch of Aorta',)
train_data_root='/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
train_ann_file='/media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/train/new1_base_novel/train_novel_10shot.json'
train_img_path='/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
val_data_root='/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
val_ann_file='/media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/val/annotation_name.json'
val_img_path='/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
test_data_root='/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
test_ann_file='/media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/test/annotation_name.json'
test_img_path='/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
class_text_path='/media/Storage3/wmm/ICML/data/texts/3VT.json'
train_batch_size_per_gpu=2
val_batch_size_per_gpu=4
test_batch_size_per_gpu=2
base_lr = 5e-5
weight_decay = 0.005



# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),

    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldDualPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                     embed_dims=text_channels,
                                     num_classes=num_training_classes),
                    ),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    test_cfg=dict(
        score_thr=0.001,  # Threshold to filter out boxes.)  
        nms=dict(type='nms', iou_threshold=0.7)
    ))


text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts','scale_factor'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),#等比例缩放
    dict(
        type='LetterResize',#图像填充
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(#这一个数据增强可以使得val上mAP50提高25个点
        type='YOLOv5RandomAffine',#随机仿射变换
        max_rotate_degree=0.0,#旋转
        max_shear_degree=0.0,#错切
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),#缩放
        max_aspect_ratio=_base_.max_aspect_ratio,#纵横比扰动：通过 max_aspect_ratio 允许在仿射变换中轻微改变宽高比（注意：这不是直接拉伸图像，而是在仿射矩阵中引入各向异性缩放）
        border_val=(114, 114, 114)),#填充色：超出边界的区域用 (114, 114, 114) 填充，与 letterbox 一致
    *_base_.last_transform[:-1],
    *text_transform
]


test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=classes),
        data_root=train_data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_path),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,
    pipeline=train_pipeline
    )

train_dataloader = dict(
    # persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=classes),
        data_root=val_data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_path),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)

coco_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=classes),
        data_root=test_data_root,
        ann_file=test_ann_file,
        test_mode=True,
        data_prefix=dict(img=test_img_path),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,
    pipeline=test_pipeline)

test_dataloader = dict(
    dataset=coco_test_dataset,
   )

val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=val_ann_file,
    metric='bbox',
    classwise=True)

test_evaluator = dict(type='mmdet.CocoMetric',
                       ann_file=test_ann_file,
                       metric='bbox',
                       classwise=True,
                    #    outfile_prefix='./work_dirs/finetune_3VT_10_shot/test_results'
                       )

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=val_interval_epochs,
    dynamic_intervals=None)


# optimizer wrapper settings
optim_wrapper = dict(
    constructor='YOLOWv5OptimizerConstructor',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),  # Use a higher learning rate for prompts
    paramwise_cfg=dict(
        bypass_duplicate=True
    )
)


# Add the custom hook
# visualizer settings
vis_backends = [dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]
# visualizer = dict(
#     type='MultiStageDetVisualizer_pro',
#     vis_backends=vis_backends,
#     name='visualizer'#虚线是矫正前的预测框
# )
visualizer = dict(
    type='mmdet.DetLocalVisualizer',       # 任务类型，如果是分类用 'ClsLocalVisualizer'，分割用 'SegLocalVisualizer'
    vis_backends=vis_backends,
    name='visualizer'
)
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    # dict(type='mmdet.DetVisualizationHook', draw=True, interval=1, show=False, score_thr=0.3)
]

default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=2,
        save_best='coco/bbox_mAP_50',
        interval=save_epoch_intervals,
        rule='greater'))

# DDP settings for frozen parameters
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)


# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage_finetune_3VT_10_shot.py 1 --amp

# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=6 bash ./tools/dist_test.sh /media/Storage3/wmm/ICML/YOLO-World-Plus/configs/config_few_shot_loss/finetune_3VT_10_shot_loss.py  /media/Storage3/wmm/ICML/YOLO-World-Plus/work_dirs/finetune_3VT_10_shot_loss/20251230_153427_Aug_0.005/best_coco_bbox_mAP_50_epoch_400.pth 1 --out work_dirs/test_results.pkl
# export PYTHONPATH="/media/Storage3/wmm/ICML/YOLO-World-Plus:$PYTHONPATH"
# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=6 bash ./tools/dist_test.sh /media/Storage3/wmm/ICML/YOLO-World-master/configs/configs_few_shot/yolo_world_l_dual_vlpan_finetune_fetal_prompt10.py /media/Storage3/wmm/ICML/YOLO-World-master/pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth 1