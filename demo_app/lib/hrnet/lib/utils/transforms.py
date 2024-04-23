from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def crop_image(img, center, scale, output_size):
    if not isinstance(center, np.ndarray):
        center = np.array(center, dtype=np.float32)
    if not isinstance(scale, np.ndarray):
        scale = np.array(scale, dtype=np.float32)
    if not isinstance(output_size, np.ndarray):
        output_size = np.array(output_size, dtype=np.float32)

    aspect_ratio = output_size[0] / output_size[1]
    w, h = scale[0] * 200, scale[1] * 200
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    # left-top, center, right-top
    src[0] = center - np.array((w / 2, h / 2), dtype=np.float32)
    # (0, 0)
    src[1] = center
    dst[1] = output_size / 2
    src[2] = center + np.array((w / 2, - h / 2), dtype=np.float32)
    dst[2] = np.array((output_size[0], 0), dtype=np.float32)

    inv_trans = cv2.getAffineTransform(dst, src)
    trans = cv2.getAffineTransform(src, dst)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )
    return dst_img, trans, inv_trans


def affine_transform(points, trans):
    points = np.concatenate([points, np.ones_like(points[..., 0:1])], axis=-1)
    points = points @ trans.T
    return points


def crop_to_heatmap(crop_size=(192, 256), heatmap_size=(48, 64)):
    heatmap_size = np.array(heatmap_size)
    crop_size = np.array(crop_size)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[1, 0] = crop_size[0]
    dst[1, 0] = heatmap_size[0]
    src[2] = crop_size // 2
    dst[2] = heatmap_size // 2
    
    inv_trans = cv2.getAffineTransform(dst, src)
    trans = cv2.getAffineTransform(src, dst)
    return trans, inv_trans