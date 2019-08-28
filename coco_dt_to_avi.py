import argparse
import cv2
import os
import numpy as np
import pycocotools.mask as mask_utils

from pycocotools.coco import COCO
from tqdm import tqdm

from utils import mask_to_polygons


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-gt", type=str, required=True)
    parser.add_argument("--coco-dt", type=str, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--iou-type", choices=["bbox", "segm"], default="segm")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--out", type=str, required=True)
    return parser


def main(args):
    coco_gt = COCO(args.coco_gt)
    coco_dt = coco_gt.loadRes(args.coco_dt)

    cat_id_to_label = {cat_id: cat["name"] for cat_id, cat in coco_dt.cats.items()}
    cat_id_to_color = {cat_id: tuple(cv2.cvtColor(np.array([[[int(255 * i / len(coco_dt.cats)), 255, 255]]], dtype=np.uint8),
                                    cv2.COLOR_HSV2RGB)[0, 0])
                       for i, cat_id in enumerate(coco_dt.cats)}
    path_to_image_id = {os.path.join(args.root, image["file_name"]): image_id
                        for image_id, image in coco_dt.imgs.items()}
    paths = sorted(path_to_image_id.keys())

    width = coco_dt.imgs[0]["width"]
    height = coco_dt.imgs[0]["height"]

    out = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        args.fps,
        (width, height)
    )

    for image_path in tqdm(paths):
        image_id = path_to_image_id[image_path]
        image_info = coco_dt.imgs[image_id]
        image_name = image_info["file_name"]
        image_path = os.path.join(args.root, image_name)
        image = cv2.imread(image_path)
        instances = np.zeros_like(image)
        for ann in coco_dt.imgToAnns[image_id]:
            cat_id = ann["category_id"]
            label = cat_id_to_label[cat_id]
            color = tuple(map(int, cat_id_to_color[cat_id]))
            if args.iou_type == "bbox":
                x, y, w, h = map(int, ann["bbox"])
                x_text = x + w
                y_text = y
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            elif args.iou_type == "segm":
                rle = ann["segmentation"]
                m = mask_utils.decode(rle)
                polygons = mask_to_polygons(m)
                polygons = [np.array(p).reshape((-1, 2)) for p in polygons]
                x_text = max([polygon[:, 0].max() for polygon in polygons])
                y_text = min([polygon[:, 1].min() for polygon in polygons])
                cv2.fillPoly(instances, polygons, color)
                cv2.polylines(image, polygons, True, color, thickness=2)
            cv2.putText(image, label, (x_text, y_text), 0, 1, color, 2)
        mask = (instances > 0).any(axis=2).astype(int)[:, :, None]
        image = (image * (1 - mask)
                 + image * mask * (1 - args.alpha)
                 + instances * mask * args.alpha).astype(np.uint8)
        out.write(image)
    out.release()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
