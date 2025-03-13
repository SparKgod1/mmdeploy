python ./tools/deploy.py `
    "D:\project\deploy\mmdeploy\configs\mmdet\panoptic-seg\panoptic-seg_maskformer_tensorrt_static-800x1344.py" `
    "D:\project\deploy\config.py" `
    "D:\project\deploy\epoch_40.pth" `
    "D:\project\deploy\demo.jpg" `
    --test-img "D:\project\deploy\demo.jpg" `
    --work-dir "D:\project\deploy\work_dir\mask2former_final1" `
    --device "cuda:0" `
    --log-level INFO `
    --show `
    --dump-info



{
    "type": "Task",
    "module": "mmdet",
    "name": "postprocess",
    "component": "ResizeInstanceMask",
    "params": {
        "is_resize_mask": true
    },
    "output": [
        "post_output"
    ], 
    "input": [
        "prep_output",
        "infer_output"
    ]
}
