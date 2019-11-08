import argparse
import cv2
import os
import numpy as np
import pycocotools.mask as mask_utils
import yaml

from pycocotools.coco import COCO
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    return parser


def load_data(annotation):
    coco_gt = COCO(annotation["GT"])
    coco_dt = coco_gt.loadRes(annotation["DT"])
    return coco_dt


def load_colors(annotation, coco_dt):
    if annotation["COLORS"] == "auto":
        cat_id_to_color = {
            cat_id: int(180 * i / len(coco_dt.cats)) for i, cat_id in enumerate(coco_dt.cats)
        }
    else:
        label_to_cat_id = {cat["name"]: cat_id for cat_id, cat in coco_dt.cats.items()}
        cat_id_to_color = {label_to_cat_id[label]: color for label, color in annotation["COLORS"].items()}
    cat_id_to_color = {
        cat_id: tuple(cv2.cvtColor(np.array([[[color, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0])
        for cat_id, color in cat_id_to_color.items()
    }
    cat_id_to_color = {cat_id: tuple(map(int, color)) for cat_id, color in cat_id_to_color.items()}
    return cat_id_to_color


def load_filenames(config, coco_dt):
    if config["ORDER"] == "name":
        order = sorted(os.listdir(config["IMAGES"]))
    elif config["ORDER"] == "id":
        ids = sorted(coco_dt.getImgIds())
        order = [coco_dt.imgs[id_]["file_name"] for id_ in ids]
    else:
        with open(config["ORDER"], "r") as f:
            order = [line.strip() for line in f]
    if config["LIMIT"] == -1:
        return order
    else:
        return order[:config["LIMIT"]]


def create_writer(config):
    writer = cv2.VideoWriter(
        config["OUT"],
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        config["FPS"],
        find_min_resolution(config)
    )
    return writer


def find_min_resolution(config):
    width, height = 0, 0
    for name, window in config["WINDOWS"].items():
        if name == "ORDER":
            continue
        p = window["POSITION"]
        r = window["RESOLUTION"]
        width = max(width, p[0] + r[0])
        height = max(height, p[1] + r[1])
    return width, height


def draw_bbox(image, anns, cat_id_to_color, cat_id_to_label, window):
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id in cat_id_to_color:
            label = cat_id_to_label[cat_id]
            color = cat_id_to_color[cat_id]

            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, window["BORDER"])
            cv2.putText(image, label, (x + w, y), 0, window["FONTSCALE"], color, window["THICKNESS"])
    return image


def draw_isegm(image, anns, cat_id_to_color, cat_id_to_label, window):
    segmentation = np.zeros_like(image)
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id in cat_id_to_color:
            label = cat_id_to_label[cat_id]
            color = cat_id_to_color[cat_id]

            rle = ann["segmentation"]
            m = mask_utils.decode(rle)
            segmentation[m.astype(bool)] = color

            x, y, w, h = map(int, ann["bbox"])
            cv2.putText(image, label, (x + w, y), 0, window["FONTSCALE"], color, window["THICKNESS"])

    image = blend(image, segmentation, window["TRANSPARENCY"])
    return image


def draw_ssegm(image, anns, cat_id_to_color, cat_id_to_label, window):
    segmentation = np.zeros_like(image)
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id in cat_id_to_color:
            color = cat_id_to_color[cat_id]
            rle = ann["segmentation"]
            m = mask_utils.decode(rle)
            segmentation[m.astype(bool)] = color

    for i, cat_id in enumerate(cat_id_to_color):
        label = cat_id_to_label[cat_id]
        color = cat_id_to_color[cat_id]
        cv2.putText(image, label, (10, (i + 1) * 30), 0, window["FONTSCALE"], color, window["THICKNESS"])

    image = blend(image, segmentation, window["TRANSPARENCY"])
    return image


def blend(image, mask, transparency):
    m = (mask > 0).any(axis=2).astype(int)[:, :, None]
    image = (image * (1 - m)
             + image * m * transparency
             + mask * m * (1 - transparency)).astype(np.uint8)
    return image


def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f)
    annotations = config["ANNOTATIONS"]

    coco_dt = {}
    for name in annotations:
        coco_dt[name] = load_data(annotations[name])

    writer = create_writer(config)

    cat_id_to_color = {}
    for name in annotations:
        cat_id_to_color[name] = load_colors(annotations[name], coco_dt[name])

    wname = list(annotations.keys())[0]
    filenames = load_filenames(config, coco_dt[wname])
    filename_to_image_id = {image["file_name"]: image_id for image_id, image in coco_dt[wname].imgs.items()}

    cat_id_to_label = {
        name: {
            cat_id: cat["name"] for cat_id, cat in coco_dt[name].cats.items()}
        for name in annotations
    }

    draw = {}
    for name in annotations:
        if annotations[name]["MODE"] == "bbox":
            draw[name] = draw_bbox
        elif annotations[name]["MODE"] == "isegm":
            draw[name] = draw_isegm
        elif annotations[name]["MODE"] == "ssegm":
            draw[name] = draw_ssegm

    width, height = find_min_resolution(config)
    for filename in tqdm(filenames):
        image_id = filename_to_image_id[filename]
        image_ = cv2.imread(os.path.join(config["IMAGES"], filename))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for wname in config["WINDOWS"]["ORDER"]:
            window = config["WINDOWS"][wname]
            x, y = window["POSITION"]
            w, h = window["RESOLUTION"]
            image = image_.copy()
            for name in window["ANNOTATIONS"]:
                anns = coco_dt[name].imgToAnns[image_id]
                frame[y:y + h, x:x + w] = \
                    draw[name](image, anns, cat_id_to_color[name], cat_id_to_label[name], annotations[name])
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
