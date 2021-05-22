_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../dataset.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            num_classes=11
        )
    )
)


optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
