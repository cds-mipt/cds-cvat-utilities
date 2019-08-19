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
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--supercat", action="store_true")
    parser.add_argument("--cvat-out", type=str, required=True)
    return parser


def main(args):
    coco_gt = COCO(args.coco_gt)
    name_to_coco_id = {img["file_name"]: coco_id for coco_id, img in coco_gt.imgs.items()}
    cat_id_to_label = {id_: coco_cat_to_label(cat, args.supercat) for id_, cat in coco_gt.cats.items()}

    cvat = CvatDataset()
    names = sorted(os.listdir(args.folder))

    for name in tqdm(names):
        ext = name.split(".")[-1]
        assert ext.lower() in ["png", "jpg", "jpeg", "tiff"], ext

        image_id = cvat.add_image()

        coco_id = name_to_coco_id[name]

        for ann in coco_gt.imgToAnns[coco_id]:
            label = cat_id_to_label[ann["category_id"]]

            if "bbox" in ann:
                x, y, width, height = ann["bbox"]
                cvat.add_box(image_id, x, y, x + width, y + height, label)

            if "segmentation" in ann:
                if type(ann["segmentation"]) == list:
                    polygons = ann["segmentation"]
                    polygons = [[[polygon[i], polygon[i + 1]] for i in range(0, len(polygon), 2)]
                                for polygon in polygons]
                else:
                    # -- pycocotools/coco.py 263:268
                    t = coco_gt.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = mask_utils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = mask_utils.decode(rle)
                    # --
                    polygons = mask_to_polygons(m)
                for polygon in polygons:
                    cvat.add_polygon(image_id, points=polygon, label=label)

    cvat.dump(args.cvat_out)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
