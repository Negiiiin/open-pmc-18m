from transformers.image_transforms import center_to_corners_format
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
import numpy as np


lab2idx = {'subfigure': 1, 'nothing': 0}
idx2lab = {v: k for k, v in lab2idx.items()}


def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


def build_dab_detr_model(checkpoint, device, idx2lab, lab2idx, image_size=480):
    image_processor = AutoImageProcessor.from_pretrained(
                checkpoint,
                do_resize=True,
                size={"width": image_size, "height": image_size},
                )
            

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=idx2lab,
        label2id=lab2idx,
        ignore_mismatched_sizes=True,
    ).to(device)
    return model, image_processor


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    Args:
        set_1 (torch.Tensor): Set 1, a tensor of dimensions (n1, 4) -- (x1, y1, x2, y2)
        set_2 (torch.Tensor): Set 2, a tensor of dimensions (n2, 4)

    Returns
    -------
        torch.Tensor: Intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    Args:
        set_1 (torch.Tensor): Set 1, a tensor of dimensions (n1, 4)
        set_2 (torch.Tensor): Set 2, a tensor of dimensions (n2, 4)

    Returns
    -------
        torch.Tensor: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    if set_1.dim() == 1 and set_1.shape[0] == 4:
        set_1 = set_1.unsqueeze(0)
    if set_2.dim() == 1 and set_2.shape[0] == 4:
        set_2 = set_2.unsqueeze(0)

    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union


def process_detections(det_boxes: torch.Tensor, det_scores: np.ndarray, nms_threshold: float):
    order = np.argsort(det_scores)
    picked_bboxes = []
    picked_scores = []
    while order.size > 0:
        index = order[-1]
        if det_boxes[index].dim() == 2 and det_boxes[index].size(0) == 1:
            picked_bboxes.append(det_boxes[index].squeeze(0).tolist())
        else:
            picked_bboxes.append(det_boxes[index].tolist())
        picked_scores.append(det_scores[index])
        if order.size == 1:
            break
        iou_with_left = (
            find_jaccard_overlap(
                det_boxes[index],
                det_boxes[order[:-1]],
            )
            .squeeze()
            .numpy()
        )

        mask = iou_with_left < nms_threshold
        order = order[:-1][mask]
    
    return picked_bboxes, picked_scores