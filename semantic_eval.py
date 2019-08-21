import argparse
import pycocotools.mask as mask_utils
import pandas as pd
import os

from collections import defaultdict
from pycocotools.coco import COCO

from utils import difference


def build_parser():
    parser = argparse.ArgumentParser("Semantic segmentation metrics")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--dt", type=str, required=True)
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


def semantic_segmentation_metrics(coco_gt, coco_dt):
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

    d = sum(T.values())
    w = {cat_id: T[cat_id] / d for cat_id in coco_gt.getCatIds()}

    metrics = defaultdict(list)
    for cat_id in coco_gt.getCatIds():
        metrics["class"].append(coco_gt.cats[cat_id]["name"])
        metrics["P"].append(P[cat_id])
        metrics["R"].append(R[cat_id])
        metrics["IoU"].append(IoU[cat_id])
        metrics["iIoU"].append(iIoU[cat_id])
        metrics["w"].append(w[cat_id])
        metrics["TP"].append(TP[cat_id])
        metrics["FP"].append(FP[cat_id])
        metrics["FN"].append(FN[cat_id])
        metrics["T"].append(T[cat_id])
        metrics["iTP"].append(iTP[cat_id])
        metrics["iFN"].append(iFN[cat_id])

    return pd.DataFrame(metrics)


def save_report(metrics, folder):
    with open(os.path.join(folder, "metrics.txt"), "w") as f:
        mIoU = metrics["IoU"].mean()
        wIoU = (metrics["IoU"] * metrics["w"]).sum()
        miIoU = metrics["iIoU"].mean()
        print("mIoU = {:.3f} | wIoU = {:.3f} | miIoU = {:.3f}".format(mIoU, wIoU, miIoU), file=f)

        for i in range(len(metrics)):
            class_ = metrics["class"].iloc[i]
            p = metrics["P"].iloc[i]
            r = metrics["R"].iloc[i]
            iou = metrics["IoU"].iloc[i]
            iiou = metrics["iIoU"].iloc[i]
            w = metrics["w"].iloc[i]
            print("class = {:>20s} w = {:.5f} P = {:.3f} R = {:.3f} IoU = {:.3f} iIoU = {:.3f}"
                  .format(class_, w, p, r, iou, iiou), file=f)


def save_csv(metrics, folder):
    path = os.path.join(folder, "metrics.csv")
    metrics.to_csv(path, index=False)


def main(args):
    os.makedirs(args.folder, exist_ok=True)

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.dt)

    metrics = semantic_segmentation_metrics(coco_gt, coco_dt)

    save_csv(metrics, args.folder)

    save_report(metrics, args.folder)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
