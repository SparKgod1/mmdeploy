# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter('mmdet.models.detectors.maskformer.'
                                     'MaskFormer.forward')
def maskformer__forward(self,
                        batch_inputs,
                        data_samples,
                        mode='tensor',
                        **kwargs):
    """Rewrite `forward` for default backend. Support configured dynamic/static
    shape for model input and return detection result as Tensor instead of
    numpy array.

    Args:
        batch_inputs (Tensor): Inputs with shape (N, C, H, W).
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        rescale (bool): Whether to rescale the results.
            Defaults to True.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            (bboxes, labels, masks, semseg), `bboxes` of shape [N, num_det, 5],
            `labels` of shape [N, num_det], `masks` of shape [N, roi_H, roi_W],
            `semseg` of shape [N, num_sem_class, sem_H, sem_W].
    """
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg

    # get origin input shape as tensor to support onnx dynamic shape
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = torch._shape_as_tensor(batch_inputs)[2:]
    if not is_dynamic_flag:
        img_shape = img_shape.to(torch.int32).tolist()
    # set the metainfo
    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')
        data_sample.set_field(
            name='batch_input_shape', value=img_shape, field_type='metainfo')

    feats = self.extract_feat(batch_inputs)
    mask_cls_results, mask_pred_results = self.panoptic_head.predict(
        feats, data_samples)

    # do not export panoptic_fusion_head
    # return mask_cls_results, mask_pred_results, torch.tensor(img_shape)


    results_list = self.panoptic_fusion_head.predict(
        mask_cls_results,
        mask_pred_results,
        data_samples,
        rescale=True)
    batch_dets = torch.zeros(len(results_list), 100, 5, dtype=torch.float32)
    batch_labels = torch.zeros(len(results_list), 100, dtype=torch.int64)
    batch_masks = torch.zeros(len(results_list), 100, 768, 1024, dtype=torch.float32)

    for i, result in enumerate(results_list):
        ins_results = result['ins_results']
        bboxes = ins_results['bboxes']
        labels = ins_results['labels']
        scores = ins_results['scores']
        masks = ins_results['masks']
        batch_dets[i, :bboxes.size(0), :bboxes.size(1)] = bboxes
        batch_dets[i, :scores.size(0), -1] = scores.view(-1)
        batch_labels[i, :labels.size(0)] = labels
        batch_masks[i, :masks.size(0), :masks.size(1)] = masks.to(torch.float32)
    return batch_dets, batch_labels, batch_masks


    ## 用以下代码就无法正常转换
    # # Step 1: Compute the maximum number of detections across all samples
    # max_detections = 0
    # for result in results_list:
    #     ins_results = result['ins_results']
    #     num_detections = ins_results['bboxes'].size(0)  # Number of bounding boxes
    #     max_detections = max(max_detections, num_detections)
    #
    # # Step 2: Allocate tensors with the dynamic size
    # batch_dets = torch.zeros(len(results_list), max_detections, 5, dtype=torch.float32)
    # batch_labels = torch.zeros(len(results_list), max_detections, dtype=torch.int64)
    # batch_masks = torch.zeros(len(results_list), max_detections, 768, 1024, dtype=torch.float32)
    #
    # # Step 3: Populate the tensors
    # for i, result in enumerate(results_list):
    #     ins_results = result['ins_results']
    #     bboxes = ins_results['bboxes']
    #     labels = ins_results['labels']
    #     scores = ins_results['scores']
    #     masks = ins_results['masks']
    #     batch_dets[i, :bboxes.size(0), :bboxes.size(1)] = bboxes
    #     batch_dets[i, :scores.size(0), -1] = scores.view(-1)
    #     batch_labels[i, :labels.size(0)] = labels
    #     batch_masks[i, :masks.size(0), :masks.size(1)] = masks.to(torch.float32)
    #
    # return batch_dets, batch_labels, batch_masks

