import argparse
import pycocotools.mask as mask_utils
import cv2

from cvat import CvatDataset
from pycocotools.coco import COCO


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-gt", type=str, required=True)
    parser.add_argument("--cvat-in", type=str, required=True)
    parser.add_argument("--cvat-out", type=str, required=True)
    return parser


def check(coco_gt, cvat):
    coco_gt_labels = coco_gt.cats.values()
    cvat_labels = cvat.get_labels()
    not_presented = set(coco_gt_labels).difference(cvat_labels)
    if len(not_presented) > 0:
        print("Labels\n{}\nnot presented in cvat but presented in coco".format(list(not_presented)))
        print("You may encounter errors during loading markup to Cvat tool")
        return False
    return True


def main(args):
    coco_gt = COCO(args.coco_gt)
    name_to_coco_id = {img["file_name"]: coco_id for coco_id, img in coco_gt.imgs.items()}

    cvat = CvatDataset()
    cvat.load(args.cvat_in)

    if not check(coco_gt, cvat):
        cmd = input("Some errors occurred. Continue anyway? (y/n)")
        if cmd == "n":
            exit()

    for image_id in cvat.get_image_ids():
        name = cvat.get_name(image_id)
        coco_id = name_to_coco_id[name]

        for ann in coco_gt.imgToAnns[coco_id]:
            label = coco_gt.cats[ann["category_id"]]

            if "bbox" in ann:
                x, y, width, height = ann["bbox"]
                cvat.add_box(image_id, x, y, x + width, y + height, label)

            if "segmentation" in ann:
                if type(ann["segmentation"]) == list:
                    polygons = ann["segmentation"]
                else:
                    # -- pycocotools/coco.py 263:268
                    t = coco_gt.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = mask_utils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = mask_utils.decode(rle)
                    # --
                    polygons, _ = cv2.findContours(m * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for polygon in polygons:
                    cvat.add_polygon(image_id, points=polygon, label=label)

    cvat.dump(args.cvat_out)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
