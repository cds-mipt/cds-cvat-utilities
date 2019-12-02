import argparse
import json
import numpy as np
import os

from cvat import CvatDataset
from utils import coco_cat_to_label


def build_parser():
    parser = argparse.ArgumentParser("Convert Cvat annotations to COCO ground truth")
    parser.add_argument("--cvat", type=str, required=True,
                        help="Cvat annotations")
    parser.add_argument("--discard", type=str, default="",
                        help="File prefix to be discarded")
    parser.add_argument("--coco-in", type=str, required=False, default=None,
                        help="Ground truth in COCO .json format")
    parser.add_argument("--coco-out", type=str, required=True,
                        help="Output file")
    parser.add_argument("--fmt", action="store_true",
                        help="Easy readable formatting")
    return parser


def main(args):
    cvat = CvatDataset()
    cvat.load(args.cvat)

    if args.coco_in is not None:
        with open(args.coco_in, "r") as f:
            coco_gt = json.load(f)
        categories = coco_gt["categories"]
        name_to_coco_id = {image["file_name"]: image["id"] for image in coco_gt["images"]}
    else:
        all_labels = []
        for image_id in cvat.get_image_ids():
            all_labels.extend(cvat.get_boxes(image_id))
            all_labels.extend(cvat.get_polygons(image_id))
        unique_labels = np.unique(np.array([i['label'] for i in all_labels]))
        categories = []
        for i, category in enumerate(unique_labels):
            categories.append({})
            categories[-1]['id'] = i+1
            categories[-1]['name'] = category
        name_to_coco_id = {os.path.relpath(cvat.get_name(image_id), args.discard): image_id
                           for image_id in cvat.get_image_ids()}
    label_to_cat_id = {coco_cat_to_label(cat): cat["id"] for cat in categories}

    images, annotations = [], []
    for image_id in cvat.get_image_ids():
        name = os.path.relpath(cvat.get_name(image_id), args.discard)
        size = cvat.get_size(image_id)

        image = {"id": image_id, "file_name": name}
        image.update(size)
        images.append(image)

        for box in cvat.get_boxes(image_id):
            x, y, width, height = box["xtl"], box["ytl"], box["xbr"] - box["xtl"], box["ybr"] - box["ytl"]
            annotation = {
                "id": len(annotations) + 1,
                "image_id": name_to_coco_id[name],
                "category_id": label_to_cat_id[box["label"]],
                "segmentation": [[x, y, x + width, y, x + width, y + height, x, y + height]],
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0,
                "conf":box["conf"]
            }
            annotations.append(annotation)

        for polygon in cvat.get_polygons(image_id):
            points = np.array(polygon["points"])
            x, y = points.min(axis=0)
            width, height = points.max(axis=0) - [x, y]
            annotation = {
                "id": len(annotations) + 1,
                "image_id": name_to_coco_id[name],
                "category_id": label_to_cat_id[polygon["label"]],
                "segmentation": [points.flatten().tolist()],
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0,
                "conf":polygon["conf"]
            }
            annotations.append(annotation)

    if args.coco_in is not None:
        images = coco_gt["images"]

    coco_gt_dct = {"images": images, "annotations": annotations, "categories": categories}
    indent = None
    separators = (",", ":")
    if args.fmt:
        indent = 2
        separators = (",", ": ")
    with open(args.coco_out, "w") as f:
        json.dump(coco_gt_dct, f, indent=indent, separators=separators)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
