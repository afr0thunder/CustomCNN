import cv2
from numpy.lib.stride_tricks import as_strided
import numpy as np

def filter_boxes(boxes, image_shape, max_boxes=6):
    if not boxes:
        return []

    h, w = image_shape[:2]
    center_x, center_y = w / 2, h / 2

    def center_dist(box):
        x, y, w, h = box
        bx = x + w / 2
        by = y + h / 2
        return np.hypot(bx - center_x, by - center_y)

    boxes_sorted = sorted(boxes, key=center_dist)
    return boxes_sorted[:max_boxes]

def make_clahe(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def mser_regions(image, config):
    temp_image = np.copy(image)
    mser = cv2.MSER_create()
    mser.setDelta(config.get("mser").get("mser_delta"))
    mser.setMinArea(config.get("mser").get("mser_min"))
    mser.setMaxArea(config.get("mser").get("mser_max"))

    if config.get("image").get("clahe"):
        temp_image = make_clahe(temp_image)

    regions, _ = mser.detectRegions(temp_image)
    boxes = []
    for reg in regions:
        x, y, w, h = cv2.boundingRect(reg)

        if w > image.shape[1] * config.get("mser").get("height_limit") or h > image.shape[0] * config.get("mser").get("width_limit"):
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < config.get("mser").get("mser_ap_min") or aspect_ratio > config.get("mser").get("mser_ap_max"):
            continue

        pad = int(config.get("mser").get("mser_pad") * max(w, h))
        x = max(x - pad, 0)
        y = max(y - pad, 0)
        w = w + 2 * pad
        h = h + 2 * pad
        boxes.append((x, y, w, h))

    return boxes

def make_pyramid(image, scale, size):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        if w < size or h < size:
            break
        image = cv2.resize(image, (w, h))
        yield image

def slide_window(image, stride, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, stride):
        for x in range(0, image.shape[1] - window_size[0] + 1, stride):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]

def extract_windows(image, window_size, stride):
    h, w = image.shape[:2]
    win_h, win_w = window_size

    shape = (
        (h - win_h) // stride + 1,
        (w - win_w) // stride + 1,
        win_h,
        win_w,
        image.shape[2] if image.ndim == 3 else 1
    )

    strides = (
        image.strides[0] * stride,
        image.strides[1] * stride,
        image.strides[0],
        image.strides[1],
        image.strides[2] if image.ndim == 3 else 1
    )

    windows = as_strided(image, shape=shape, strides=strides)
    windows = windows.reshape(-1, win_h, win_w, image.shape[2] if image.ndim == 3 else 1)
    return windows

def sliding_regions(image, config):
    stride = config["window"]["stride"]
    scale = config["pyramid"]["pyramid_scale"]
    window_sizes = config["window"]["window_size"]

    boxes = []
    for size in window_sizes:
        for scaled in make_pyramid(image, scale, size):
            scale_factor = image.shape[0] / scaled.shape[0]
            h, w = scaled.shape[:2]
            for y in range(0, h - size + 1, stride):
                for x in range(0, w - size + 1, stride):
                    x_orig = int(x * scale_factor)
                    y_orig = int(y * scale_factor)
                    w_orig = int(size * scale_factor)
                    h_orig = int(size * scale_factor)
                    boxes.append((x_orig, y_orig, w_orig, h_orig))
    return boxes

def nms(boxes, threshold=0.3):
    if not boxes:
        return []

    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = x1 + boxes_np[:, 2]
    y2 = y1 + boxes_np[:, 3]
    areas = boxes_np[:, 2] * boxes_np[:, 3]
    scores = areas
    idxs = np.argsort(-scores)

    pick = []
    while len(idxs) > 0:
        current = idxs[0]
        pick.append(current)
        xx1 = np.maximum(x1[current], x1[idxs[1:]])
        yy1 = np.maximum(y1[current], y1[idxs[1:]])
        xx2 = np.minimum(x2[current], x2[idxs[1:]])
        yy2 = np.minimum(y2[current], y2[idxs[1:]])
        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        iou = inter / (areas[current] + areas[idxs[1:]] - inter + 1e-6)
        idxs = idxs[np.where(iou <= threshold)[0] + 1]

    return [tuple(boxes_np[i]) for i in pick]

def detect_boxes(image, config):
    boxes = []

    if config.get("mser").get("use_mser"):
        mser_boxes = mser_regions(image, config)
        boxes.extend(mser_boxes)

    if config.get("window").get("sliding_window"):
        sw_boxes = sliding_regions(image, config)
        boxes.extend(sw_boxes)

    if config.get("nms").get("nms_apply"):
        threshold = config.get("nms").get("threshold")
        boxes = nms(boxes, threshold)

    boxes = filter_boxes(boxes, image.shape)
    return boxes