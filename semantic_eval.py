import argparse
import pycocotools.mask as mask_utils

from collections import defaultdict
from pycocotools.coco import COCO

from utils import difference


def build_parser():
    parser = argparse.ArgumentParser("Semantic segmentation metrics")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--dt", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["class", "category"], default="class")
    parser.add_argument("--folder", type=str, default="results")
    return parser


def merge_annotations(coco):
    res = defaultdict(list)
    for ann in coco.anns.values():
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        rle = coco.annToRLE(ann)
        res[image_id, cat_id].append(rle)
    for k, rles in res.items():
        res[k] = mask_utils.merge(rles)
    return res


def main(args):
    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.dt)

    gt = merge_annotations(coco_gt)
    dt = merge_annotations(coco_dt)

    total_area, count = defaultdict(int), defaultdict(int)
    for (image_id, cat_id), rle in gt.items():
        total_area[cat_id] += mask_utils.area([rle])[0]
        count[cat_id] += 1

    mean_area = defaultdict(int)
    for k in total_area:
        mean_area[k] = total_area[k] / count[k]

    TP, FP, FN = defaultdict(int), defaultdict(int), defaultdict(int)
    T = defaultdict(int)
    iTP, iFN = defaultdict(int), defaultdict(int)
    for cat_id in coco_gt.getCatIds():
        for image_id in coco_gt.getImgIds():
            k = image_id, cat_id
            if k in gt and k in dt:
                gt_area = mask_utils.area([gt[k]])[0]
                iou = mask_utils.iou([dt[k]], [gt[k]], [False])[0, 0]
                u = mask_utils.area([mask_utils.merge([dt[k], gt[k]])])[0]
                i = iou * u
                TP[cat_id] += i
                iTP[cat_id] += i * mean_area[cat_id] / gt_area
                FP[cat_id] += mask_utils.area([difference(dt[k], gt[k])])[0]
                fn = mask_utils.area([difference(gt[k], dt[k])])[0]
                FN[cat_id] += fn
                iFN[cat_id] += fn * mean_area[cat_id] / gt_area
            elif k in gt:
                gt_area = mask_utils.area([gt[k]])[0]
                FN[cat_id] += gt_area
                iFN[cat_id] += mean_area[cat_id]
            elif k in dt:
                FP[cat_id] += mask_utils.area([dt[k]])[0]

            if k in gt:
                T[cat_id] += mask_utils.area([gt[k]])[0]

    P = {cat_id: TP[cat_id] / (TP[cat_id] + FP[cat_id]) for cat_id in coco_gt.getCatIds()}
    R = {cat_id: TP[cat_id] / (TP[cat_id] + FN[cat_id]) for cat_id in coco_gt.getCatIds()}

    IoU = {cat_id: TP[cat_id] / (TP[cat_id] + FP[cat_id] + FN[cat_id]) for cat_id in coco_gt.getCatIds()}
    iIoU = {cat_id: iTP[cat_id] / (iTP[cat_id] + FP[cat_id] + iFN[cat_id]) for cat_id in coco_gt.getCatIds()}

    mIoU = sum(IoU.values()) / len(IoU)
    miIoU = sum(iIoU.values()) / len(iIoU)

    d = sum(T.values())
    fwIoU = sum(T[cat_id] * IoU[cat_id] / d for cat_id in coco_gt.getCatIds())

    print("mIoU = {:.3f} | f.w. IoU = {:.3f} | miIoU = {:.3f}".format(mIoU, fwIoU, miIoU))

    for cat_id in coco_gt.getCatIds():
        cat = coco_gt.cats[cat_id]
        p = P[cat_id]
        r = R[cat_id]
        iou = IoU[cat_id]
        iiou = iIoU[cat_id]
        w = T[cat_id] / d
        print("class = {} P = {:.3f} R = {:.3f} IoU = {:.3f} iIoU = {:.3f} w = {:.5f}".format(cat["name"], p, r, iou, iiou, w))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
