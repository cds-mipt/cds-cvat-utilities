import cv2
import numpy as np
import pycocotools.mask as mask_utils


def coco_cat_to_label(cat, supercategory=False):
    label = cat["name"]
    if supercategory:
        label = cat["supercategory"] + "/" + cat["name"]
    label = label.replace(" ", "_")
    return label


def mask_to_polygons(m):
    polygons, _ = cv2.findContours(m * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = [[point[0].tolist() for point in polygon] for polygon in polygons]
    return polygons


def clip_point(x, y, W, H):
    x = np.clip(x, 0, W)
    y = np.clip(y, 0, H)
    return x, y


def invert(rleObjs):
    masks = mask_utils.decode(rleObjs)
    masks = 1 - masks
    return mask_utils.encode(masks)


def intersect(rleObjs):
    rleObjs = invert(rleObjs)
    i = mask_utils.merge(rleObjs)
    return invert([i])[0]


def difference(a, b):
    ib = invert([b])[0]
    return intersect([a, ib])
