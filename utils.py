import cv2
import numpy as np


def coco_cat_to_label(cat, supercategory=False):
    label = cat["name"]
    if supercategory:
        label = cat["supercategory"] + "/" + cat["name"]
    label = label.replace(" ", "_")
    return label


def mask_to_polygons(m):
    polygons, _ = cv2.findContours(m * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [[point[0].tolist() for point in polygon] for polygon in polygons]
    return polygons


def clip_point(x, y, W, H):
    x = np.clip(x, 0, W)
    y = np.clip(y, 0, H)
    return x, y
