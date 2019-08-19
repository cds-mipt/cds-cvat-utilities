import argparse
import pycocotools.mask as mask_utils
import os

from pycocotools.coco import COCO
from tqdm import tqdm

from cvat import CvatDataset
from utils import coco_cat_to_label, mask_to_polygons


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-gt", type=str, required=True)
    parser.add_argument("--coco-dt", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--supercat", action="store_true")
    parser.add_argument("--iou-type", type=str, choices=["segm", "bbox"], default="segm")
    parser.add_argument("--cvat-out", type=str, required=True)
    return parser


def main(args):
    coco_gt = COCO(args.coco_gt)
    coco_dt = coco_gt.loadRes(args.coco_dt)
    coco_id_to_name = {coco_id: img["file_name"] for coco_id, img in coco_gt.imgs.items()}
    cat_id_to_label = {id_: coco_cat_to_label(cat, args.supercat) for id_, cat in coco_gt.cats.items()}

    name_to_cvat_id = {name: id_ for id_, name in enumerate(sorted(os.listdir(args.folder)))}

    cvat = CvatDataset()
    for _ in range(len(name_to_cvat_id)):
        cvat.add_image()

    for ann in tqdm(coco_dt.anns.values()):
        coco_id = ann["image_id"]
        cvat_id = name_to_cvat_id[coco_id_to_name[coco_id]]
        label = cat_id_to_label[ann["category_id"]]
        conf = ann["score"]
        if args.iou_type == "bbox":
            x, y, width, height = ann["bbox"]
            cvat.add_box(cvat_id, x, y, x + width, y + height, label, conf=conf)
        elif args.iou_type == "segm":
            rle = ann["segmentation"]
            m = mask_utils.decode(rle)
            polygons = mask_to_polygons(m)
            for polygon in polygons:
                cvat.add_polygon(cvat_id, points=polygon, label=label, conf=conf)

    cvat.dump(args.cvat_out)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
