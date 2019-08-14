import argparse
import pycocotools.mask as mask_utils
import cv2

from cvat import CvatDataset
from pycocotools.coco import COCO


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-gt", type=str, required=True)
    parser.add_argument("--coco-dt", type=str, required=True)
    parser.add_argument("--cvat-in", type=str, required=True)
    parser.add_argument("--cvat-out", type=str, required=True)
    return parser


def main(args):
    coco_gt = COCO(args.coco_gt)
    coco_dt = coco_gt.loadRes(args.coco_dt)
    coco_id_to_name = {coco_id: img["file_name"] for coco_id, img in coco_gt.imgs.items()}

    cvat = CvatDataset()
    cvat.load(args.cvat_in)
    name_to_cvat_id = {image_id: cvat.get_name(image_id) for image_id in cvat.get_image_ids()}

    for ann in coco_dt.anns:
        coco_id = ann["image_id"]
        cvat_id = name_to_cvat_id[coco_id_to_name[coco_id]]
        label = coco_gt.cats[ann["category_id"]]
        conf = ann["score"]
        if "bbox" in ann:
            x, y, width, height = ann["bbox"]
            cvat.add_box(cvat_id, x, y, x + width, y + height, label, conf=conf)
        if "segmentation" in ann:
            rle = ann["segmentation"]
            m = mask_utils.decode(rle)
            polygons, _ = cv2.findContours(m * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for polygon in polygons:
                cvat.add_polygon(cvat_id, points=polygon, label=label)

    cvat.dump(args.cvat_out)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
