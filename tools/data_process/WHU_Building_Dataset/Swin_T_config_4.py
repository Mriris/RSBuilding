backbone_checkpoint = None
base_lr = 0.0001 # Change if you want to
bs = 2 # Change if you want to
bs_mult = 1 # Change if you want to
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='FoundationInputSegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=200, type='CheckpointHook'), # Change if you want to
    logger=dict(interval=20, log_metric_by_epoch=False, type='LoggerHook'), # Change if you want to
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        img_shape=(
            512,
            512,
            3,
        ), interval=1, type='CDVisualizationHook')) # Change if you want to
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
finetune_cfg = [
    'neck',
    'decoder',
    'ab mask head',
    'cd mask head',
    'ab query head',
    'cd query head',
]
gpu_nums = 8 # Change if you want to
launcher = 'none'
load_from = '/home/iris/PycharmProjects/RSBuilding/pretrain_ckpt/Swin_T.pth' ############### Must be changed. Path to your pretrained model
log_level = 'INFO'
log_processor = dict(by_epoch=False)
logger_interval = 20 # Change if you want to
max_iters = 40000.0 # Change if you want to
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.15,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=None,
        mlp_ratio=4,
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        patch_size=4, # Change if you want to
        pretrain_img_size=224,
        qk_scale=None,
        qkv_bias=True,
        strides=(
            4,
            2,
            2,
            2,
        ),
        type='mmseg.SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='FoundationInputSegDataPreProcessor'),
    decode_head=dict(
        drop=0.0,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        loss_type='BCELoss',
        loss_weight=[
            1,
            1,
            1,
        ],
        out_channels=256,
        type='Foundation_Decoder_swin_v1'),
    finetune_cfg=[
        'neck',
        'decoder',
        'ab mask head',
        'cd mask head',
        'ab query head',
        'cd query head',
    ],
    test_cfg=dict(),
    train_cfg=dict(),
    type='FoundationEncoderDecoder')
names = 'Swin_T_whu'
num_workers = 8 # Change if you want to
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05), # Change if you want to
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.01),
            building_a_embed=dict(decay_mult=0.0, lr_mult=1.0),
            building_b_embed=dict(decay_mult=0.0, lr_mult=1.0),
            cd_embed=dict(decay_mult=0.0, lr_mult=1.0)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=40000.0,
        eta_min=1e-07,
        power=1.0,
        type='PolyLR'),
] # Change if you want to
persistent_workers = True
resume = False
resume_from = None
test_cfg = dict(type='TestLoop')
test_data_list = '/home/iris/PycharmProjects/RSBuilding/data_list/levircd/test.txt' ########## Check whether it right or not
test_dataloader = dict(
    batch_sampler=dict(type='BatchSampler_Modified'),
    batch_size=4,
    dataset=dict(
        data_list='/home/iris/PycharmProjects/RSBuilding/data_list/levircd/test.txt', ############ Check whether it right or not
        pipeline=[
            dict(type='MultiImgLoadImageFromFile_Modified'),
            dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
            dict(type='MultiImgPackSegInputs_Modified'),
        ],
        type='FoundationDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='IoU_Base_Metric_Modified')
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile_Modified'),
    dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
    dict(type='MultiImgPackSegInputs_Modified'),
]
train_cfg = dict(
    max_iters=40000.0, type='IterBasedTrainLoop', val_interval=200)
train_data_list = '/home/iris/PycharmProjects/RSBuilding/data_list/mua2/train.txt' ############ Check it
train_dataloader = dict(
    batch_sampler=dict(type='BatchSampler_Modified'),
    batch_size=2,
    dataset=dict(
        dataset=dict(
            backend_args=None,
            data_list='/home/iris/PycharmProjects/RSBuilding/data_list/mua2/train.txt', ########### Check it
            pipeline=[
                dict(type='MultiImgLoadImageFromFile_Modified'),
                dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
                dict(
                    cat_max_ratio=0.95,
                    crop_size=(
                        512,
                        512,
                    ),
                    type='MultiImgRandomCrop_Modified'),
                dict(
                    direction='horizontal',
                    prob=0.5,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
                dict(prob=0.5, type='MultiImgExchangeTime'),
                dict(
                    brightness_delta=10,
                    contrast_range=(
                        0.8,
                        1.2,
                    ),
                    hue_delta=10,
                    saturation_range=(
                        0.8,
                        1.2,
                    ),
                    type='MultiImgPhotoMetricDistortion'),
            ],
            type='FoundationDataset'),
        pipeline=[
            dict(
                center_ratio_range=(
                    0.25,
                    0.75,
                ),
                img_scale=(
                    512,
                    512,
                ),
                prob=0.5,
                type='RandomMosaic_Modified'),
            dict(type='MultiImgPackSegInputs_Modified'),
        ],
        type='MultiImageMixDataset_Modified'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_dataset = dict(
    dataset=dict(
        backend_args=None,
        data_list='/home/iris/PycharmProjects/RSBuilding/data_list/mua2/train.txt', ########### Check it
        pipeline=[
            dict(type='MultiImgLoadImageFromFile_Modified'),
            dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
            dict(
                cat_max_ratio=0.95,
                crop_size=(
                    512,
                    512,
                ),
                type='MultiImgRandomCrop_Modified'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(prob=0.5, type='MultiImgExchangeTime'),
            dict(
                brightness_delta=10,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='MultiImgPhotoMetricDistortion'),
        ],
        type='FoundationDataset'),
    pipeline=[
        dict(
            center_ratio_range=(
                0.25,
                0.75,
            ),
            img_scale=(
                512,
                512,
            ),
            prob=0.5,
            type='RandomMosaic_Modified'),
        dict(type='MultiImgPackSegInputs_Modified'),
    ],
    type='MultiImageMixDataset_Modified')
train_pipeline = [
    dict(
        center_ratio_range=(
            0.25,
            0.75,
        ),
        img_scale=(
            512,
            512,
        ),
        prob=0.5,
        type='RandomMosaic_Modified'),
    dict(type='MultiImgPackSegInputs_Modified'),
]
tta_model = dict(type='mmseg.SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_sampler=dict(type='BatchSampler_Modified'),
    batch_size=2,
    dataset=dict(
        data_list='/home/iris/PycharmProjects/RSBuilding/data_list/levircd/test.txt', ############ Check it
        pipeline=[
            dict(type='MultiImgLoadImageFromFile_Modified'),
            dict(type='MultiImgMultiAnnLoadAnnotations_Modified'),
            dict(type='MultiImgPackSegInputs_Modified'),
        ],
        type='FoundationDataset'),
    num_workers=8, # Change if you want to
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='IoU_Base_Metric_Modified')
val_interval = 200
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
wandb = 0
work_dir = '/home/iris/PycharmProjects/RSBuilding/saved_models/mua4' ########## Change to folder where you want to save your checkpoint when training