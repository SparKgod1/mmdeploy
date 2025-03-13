_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/tensorrt.py'
]
onnx_config = dict(
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
    ])
