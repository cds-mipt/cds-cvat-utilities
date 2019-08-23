import argparse
import json
import pycocotools.mask as mask_utils

from pycocotools.coco import COCO
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser("Convert COCO annotations to predictions")
    parser.add_argument("--coco-gt", type=str, required=True,
                        help="Ground truth in COCO .json format")
    parser.add_argument("--coco-dt", type=str, required=True,
                        help="Predictions in COCO .json format")
    parser.add_argument("--iou-type", type=str, choices=["segm", "bbox"], default="segm",
                        help="Use polygon/bbox in predictions")
    parser.add_argument("--fmt", action="store_true",
                        help="Easy readable formatting")
    return parser


def main(args):
    coco_gt = COCO(args.coco_gt)

    coco_dt_lst = []
    for ann in tqdm(coco_gt.anns.values()):
        annotation = {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "score": 1.0
        }
        if args.iou_type == "segm":
            rle = mask_utils.encode(coco_gt.annToMask(ann))
            rle['counts'] = rle['counts'].decode('ascii')
            annotation["segmentation"] = rle
        elif args.iou_type == "bbox":
            annotation["bbox"] = ann["bbox"]
        coco_dt_lst.append(annotation)

    indent = None
    separators = (",", ":")
    if args.fmt:
        indent = 2
        separators = (",", ": ")
    with open(args.coco_dt, "w") as f:
        json.dump(coco_dt_lst, f, indent=indent, separators=separators)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
