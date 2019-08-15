import argparse
import json
import numpy as np

from cvat import CvatDataset


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat", type=str, required=True)
    parser.add_argument("--coco-in", type=str, required=False, default=None)
    parser.add_argument("--coco-out", type=str, required=True)
    parser.add_argument("--fmt", action="store_true")
    return parser


def main(args):
    if args.coco_in is not None:
        raise NotImplementedError

    cvat = CvatDataset()
    cvat.load(args.cvat)

    label_to_cat_id = {label: i + 1 for i, label in enumerate(cvat.get_labels())}
    categories = [{"id": cat_id, "name": label, "supercategory": "entity"}
                  for label, cat_id in label_to_cat_id.items()]

    images, annotations = [], []
    for image_id in cvat.get_image_ids():
        name = cvat.get_name(image_id)
        size = cvat.get_size(image_id)

        image = {"id": image_id, "file_name": name}
        image.update(size)
        images.append(image)

        for box in cvat.get_boxes(image_id):
            x, y, width, height = box["xtl"], box["ytl"], box["xbr"] - box["xtl"], box["ybr"] - box["ytl"]
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": label_to_cat_id[box["label"]],
                "segmentation": [[x, y, x + width, y, x + width, y + height, x, y + height]],
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0
            }
            annotations.append(annotation)

        for polygon in cvat.get_polygons(image_id):
            points = np.array(polygon["points"])
            x, y = points.min(axis=0)
            width, height = points.max(axis=0) - [x, y]
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": label_to_cat_id[polygon["label"]],
                "segmentation": points.flatten().tolist(),
                "area": width * height,
                "bbox": [x, y, width, height],
                "iscrowd": 0
            }
            annotations.append(annotation)

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
