import numpy as np
import gc
import cv2

def keep_masks(all_frames_masks, min_len, max_len):
    keep = [
        i for i, m in enumerate(all_frames_masks)
        if min_len < m['bbox'][2] < max_len and min_len < m['bbox'][3] < max_len
    ]
    return [all_frames_masks[j] for j in keep]

def xywh_to_xyxy(bbox):
    return [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]

def bboxes_from_masks(masks, image_shape, buffer):
    bounding_boxes = []
    for m in masks:
        x_min, y_min, x_max, y_max = xywh_to_xyxy(m['bbox'])
        bounding_boxes.append([
            max(0, x_min - buffer),
            max(0, y_min - buffer),
            min(x_max + buffer, image_shape[1]),
            min(y_max + buffer, image_shape[0])
        ])
    return bounding_boxes

def get_bounding_box(mask):
    mask = np.squeeze(mask)
    y_idx, x_idx = np.where(mask)
    if not x_idx.size or not y_idx.size:
        return None
    return np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)

def calculate_iou(b1, b2):
    xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0

def mask_to_polygons(mask):
    mask = np.squeeze(mask.astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)
    return polygons

def polygons_to_mask(polygons, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)



