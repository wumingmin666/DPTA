# Stage 3: Joint Training with Anatomy-Aware & Prototype Calibration
# 基于 stage1_finetune_3VT_10_shot.py 修改

_base_ = (
    '../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
default_scope = 'mmyolo'
# ==============================================================================
# 1. 导入自定义模块 (确保这些路径下的 __init__.py 已经注册了类)
# ==============================================================================
custom_imports = dict(
    imports=[
        'med_yolo.models.dense_heads.med_head', 
        'med_yolo.models.losses.topology_loss',
        'med_yolo.models.losses.contrastive_loss'
    ],
    allow_failed_imports=False)

# ==============================================================================
# 2. 关键路径设置 (请根据你的实际环境修改)
# ==============================================================================
# 【重要】Stage 3 必须加载 Stage 1 训练好的最佳权重，而不是原始的 YOLO-World
stage1_checkpoint = '/media/Storage3/wmm/ICML/Medical-OD/checkpoints/best_coco_bbox_mAP_50_epoch_380.pth' 

# 【重要】Stage 2 生成的原型库文件路径
prototype_path = 'work_dirs/prototypes.pt' 

# 沿用 Stage 1 的数据路径
data_root = '/media/Storage3/wmm/ICML/data/Fetus/Hospital_1/three_vessel_tracheal'
train_ann_file = '/media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/train/new1_base_novel/train_novel_10shot.json'
val_ann_file = '/media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/val/annotation_name.json'
test_ann_file = '/media/Storage3/wmm/ICML/data/fetus_annotations_coco/3VT/c1/test/annotation_name.json'
class_text_path = '/media/Storage3/wmm/ICML/data/texts/3VT.json'
text_model_path = '/media/Storage3/wmm/ICML/YOLO-World-master/pretrained_models/clip-vit-base-patch32'
classes=('Ascending Aorta','Spine','Pulmonary trunk & ductus arteriosus','Trachea','Superior vena cava','Arch of Aorta',)
# ==============================================================================
# 3. 基础参数
# ==============================================================================
num_classes = 6
num_training_classes = 6
text_channels = 512
# YOLOv8-L 的 Neck 输出配置，保持与 Stage 1 一致
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

# 训练参数
max_epochs = 200     # 10-shot 收敛较快，不需要 6000 epoch
base_lr = 1e-4       # 微调阶段学习率适当降低
weight_decay = 0.05
train_batch_size_per_gpu = 4 # 如果显存允许，稍微大一点有利于拓扑关系的计算
load_from = stage1_checkpoint 

val_interval_epochs=10
save_epoch_intervals = 10
# ==============================================================================
# 4. 模型配置 (核心修改)
# ==============================================================================
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),

    # --- Backbone ---
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        # 【生存策略】冻结 Backbone，保护 Stage 1 的领域适应特征
        # frozen_stages=4 表示冻结所有 Stage
        frozen_stages=4, 
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_path,
            frozen_modules=['all'])),

    # --- Neck ---
    neck=dict(type='YOLOWorldDualPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),

    # --- Head (替换为 UltrasoundYOLOHead) ---
    bbox_head=dict(
        # _delete_=True, # 删除原有的 Head 配置
        type='UltrasoundYOLOHead',
        
        # 基础 Head Module 配置 (复用卷积层定义)
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=text_channels, # 512
            num_classes=num_training_classes,
            # use_bn_head=True
        ),
        
        # 自定义路径与参数
        prototype_path=prototype_path,
        vis_input_dim=text_channels, # 512, 必须与 Neck 输出一致
        geo_input_dim=4,
        geo_embed_dim=64,
        final_embed_dim=64,
        
        # Loss 权重配置
        loss_iou_branch=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_contrastive=dict(type='PrototypeContrastiveLoss', loss_weight=0.5),
        loss_topology=dict(type='TopologyEnergyLoss', loss_weight=2.0), # 拓扑约束核心
        
        # 基础 Loss (复用官方 YOLOv8 参数)
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='none', loss_weight=0.5),
        loss_bbox=dict(type='IoULoss', iou_mode='ciou', reduction='none', loss_weight=7.5),
        loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='none', loss_weight=1.5),
    ),

    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    test_cfg=dict(
        score_thr=0.001,
        # 测试时的 NMS 配置
        nms=dict(type='nms', iou_threshold=0.5) 
    ))

# ==============================================================================
# 5. 数据增强 (Pipeline)
# ==============================================================================
# 文本变换
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

# 训练变换
train_pipeline = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    
    # 【修改点】数据增强策略调整
    # 移除了 RandomFlip (随机翻转)，因为解剖结构的左右关系是固定的 (如心脏朝向)
    # 翻转会导致 L_energy (拓扑损失) 学习到错误的相对位置
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=5.0,   # 轻微旋转是可以的
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
        
    *_base_.last_transform[:-1],
    *text_transform
]

# 测试变换 (保持不变)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# ==============================================================================
# 6. 数据集加载
# ==============================================================================
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=classes), # classes 变量来自 Stage 1 定义 (3VT 类别)
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=data_root), # 注意：这里如果 data_root 就是图片根目录
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)

# ... (Val 和 Test Dataset 保持与 Stage 1 一致，略微省略以节省篇幅，直接复用即可) ...
# 注意：你需要把 stage1 中的 coco_val_dataset, coco_test_dataset, val_dataloader, test_dataloader 
# 以及 val_evaluator, test_evaluator 完整复制过来。
# 所有的 data_root 和 ann_file 保持不变。

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=data_root),
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
        data_root=data_root,
        ann_file=test_ann_file,
        test_mode=True,
        data_prefix=dict(img=data_root),
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

# ==============================================================================
# 7. 优化器与 Hooks
# ==============================================================================
optim_wrapper = dict(
    constructor='YOLOWv5OptimizerConstructor',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        bypass_duplicate=True
    )
)
vis_backends = [dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',       # 任务类型，如果是分类用 'ClsLocalVisualizer'，分割用 'SegLocalVisualizer'
    vis_backends=vis_backends,
    name='visualizer'
)
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=2,
        save_best='coco/bbox_mAP_50',
        interval=save_epoch_intervals, # 稍微频繁一点保存
        rule='greater'),
    visualization=dict(type='mmdet.DetVisualizationHook', draw=True, interval=1, score_thr=0.05)
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# 分布式设置
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True # 必须开启，因为我们冻结了 Backbone，部分参数不参与反向传播
)


# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 bash ./tools/dist_train.sh /media/Storage3/wmm/ICML/Medical-OD/configs/stage3_finetune_3VT_10_shot.py 1 --amp

# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 bash ./tools/dist_test.sh /media/Storage3/wmm/ICML/YOLO-World-Plus/configs/config_few_shot_GNN/finetune_3VT_10_shot.py  /media/Storage3/wmm/ICML/YOLO-World-Plus/work_dirs/finetune_3VT_10_shot/best_coco_bbox_mAP_50_epoch_800.pth 1 --out work_dirs/test_results.pkl
# export PYTHONPATH="/media/Storage3/wmm/ICML/YOLO-World-Plus:$PYTHONPATH"