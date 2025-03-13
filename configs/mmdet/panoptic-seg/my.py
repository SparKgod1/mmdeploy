_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/tensorrt.py'
]
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
        'masks': {
            0: 'batch',
            1: 'num_dets',
            2: 'height',
            3: 'width'
        },
    },
    opset_version=13,
    output_names=['dets', 'labels', 'masks'],
    input_shape=[1024, 768])

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 768, 1024],
                    opt_shape=[1, 3, 768, 1024],
                    max_shape=[1, 3, 768, 1024])))
    ],
    model_outputs=[
        dict(
            output_shapes=dict(
                dets=dict(
                    min_shape=[1, 100, 5],
                    opt_shape=[1, 100, 5],
                    max_shape=[1, 100, 5]
                ),
                labels=dict(
                    min_shape=[1, 100],
                    opt_shape=[1, 100],
                    max_shape=[1, 100]
                ),
                masks=dict(
                    min_shape=[1, 100, 768, 1024],  # H=1024, W=768
                    opt_shape=[1, 100, 768, 1024],
                    max_shape=[1, 100, 768, 1024]
                )
            )
        )
    ]
)
