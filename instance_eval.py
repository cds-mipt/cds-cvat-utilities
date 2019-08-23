import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


np.warnings.filterwarnings("ignore")


PR_CURVES = [
    # {"iouThr": 0.5, "area": "all", "class": "person", "maxDet": 10},
    # {"iouThr": 0.75, "area": "all", "class": "person", "maxDet": 10}
]


class Params:

    def __init__(self, gt, iouType):
        """
        iouType - one of 'bbox', 'segm'
        """
        # список id изображений для подсчета метрик
        # пустой - использовать все
        self.imgIds = []

        self.classes = []

        # пороги IoU
        self.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        # площади объектов, для которых будут вычислeны метрики
        self.areas = {
            "all": [0 ** 2, 1e5 ** 2],
            "small": [0 ** 2, 32 ** 2],
            "medium": [32 ** 2, 96 ** 2],
            "large": [96 ** 2, 1e5 ** 2]
        }

        self.maxDets = [1, 10, 100]

        # остальное, как правило, нет причин менять
        self.id_to_class = {cat_id: cat["name"] for cat_id, cat in gt.cats.items()}

        self.class_to_id = {cat["name"]: cat_id for cat_id, cat in gt.cats.items()}
        self.catIds = [self.class_to_id[cls] for cls in self.classes] or list(gt.cats.keys())
        self.useCats = 1
        self.iouType = iouType
        self.useSegm = None
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.areaRngLbl = list(self.areas.keys())
        self.areaRng = [self.areas[k] for k in self.areaRngLbl]
        if not self.imgIds:
            self.imgIds = sorted(gt.getImgIds())


def build_parser():
    parser = argparse.ArgumentParser("Instance segmentation and object detection metrics")
    parser.add_argument("--gt", type=str, required=True,
                        help="Ground truth in COCO .json format")
    parser.add_argument("--dt", type=str, required=True,
                        help="Predictions in COCO .json format")
    parser.add_argument("--iou-type", type=str, default="segm", choices=["bbox", "segm"],
                        help="Detection/segmentation metrics")
    parser.add_argument("--score-thr", type=float, default=0.,
                        help="Filter predictions by score")
    parser.add_argument("--folder", type=str, default="results",
                        help="Place evaluation results here")
    return parser


def detection_metrics(coco_gt, coco_dt, params):
    def calk_cond_mean(s, area, cat_id=-1, iouThr="mean", maxDet=-1):
        p = coco_eval.params
        s = s[:, :, list(p.areaRngLbl).index(area), p.maxDets.index(maxDet)]
        if cat_id != -1:
            s = s[:, p.catIds.index(cat_id)]
        if iouThr != "mean":
            s = s[list(p.iouThrs).index(iouThr)]
        valid = s > -1
        return np.mean(s[valid]) if valid.any() else -1

    def AP(area, cat_id=-1, iouThr=None, maxDet=-1):
        s = coco_eval.eval['precision'].mean(axis=1)
        return calk_cond_mean(s, area, cat_id, iouThr, maxDet)

    def AR(area, cat_id=-1, iouThr=None, maxDet=-1):
        s = coco_eval.eval['recall']
        return calk_cond_mean(s, area, cat_id, iouThr, maxDet)

    def pr_curve(area, cat_id, iouThr, maxDet):
        p = coco_eval.params
        recall = p.recThrs
        ti = list(p.iouThrs).index(iouThr)
        ki = list(p.catIds).index(cat_id)
        ai = list(p.areaRngLbl).index(area)
        di = list(p.maxDets).index(maxDet)
        precision = coco_eval.eval['precision'][ti, :, ki, ai, di]
        return recall, precision

    coco_eval = COCOeval(coco_gt, coco_dt, params.iouType)
    coco_eval.params = params
    coco_eval.evaluate()
    coco_eval.accumulate()

    metrics = []
    p = coco_eval.params
    for cat_id in p.catIds:
        for area in p.areaRngLbl:
            for maxDet in p.maxDets:
                for iouThr in p.iouThrs:
                    ap = AP(area, cat_id, iouThr, maxDet)
                    ar = AR(area, cat_id, iouThr, maxDet)
                    recall, precision = pr_curve(area, cat_id, iouThr, maxDet)
                    metrics.append({
                        "class": p.id_to_class[cat_id],
                        "area": area,
                        "maxDet": maxDet,
                        "iouThr": iouThr,
                        "AP": ap,
                        "AR": ar,
                        "recall": list(recall),
                        "precision": list(precision)
                    })

    return pd.DataFrame(metrics)


def save_csv(metrics, folder):
    path = os.path.join(folder, "metrics.csv")
    metrics.to_csv(path, index=False)


def save_report(metrics, folder=None):
    f = None
    if folder is not None:
        f = open(os.path.join(folder, "metrics.txt"), "w")

    area_list = sorted(set(metrics["area"]))
    maxDet_list = sorted(set(metrics["maxDet"]))
    iouThr_list = sorted(set(metrics["iouThr"]))

    mean_msg = "[area = {:6s} | IoU = {:<4} | maxDets = {:<3} ]  mAP = {:0.3f}  mAR = {:0.3f}"
    indexed = metrics.set_index(["area", "maxDet"])
    for area in area_list:
        for maxDet in maxDet_list:
            sdf = indexed.loc[(area, maxDet)]
            mAP, mAR = sdf["AP"].mean(), sdf["AR"].mean()
            print(mean_msg.format(area, "mean", maxDet, mAP, mAR), file=f)

            sdf = sdf.reset_index().set_index(["area", "maxDet", "iouThr"])
            for iouThr in iouThr_list:
                ssdf = sdf.loc[(area, maxDet, iouThr)]
                mAP, mAR = ssdf["AP"].mean(), ssdf["AR"].mean()
                print(mean_msg.format(area, iouThr, maxDet, mAP, mAR), file=f)
            print(file=f)

    if f is not None:
        f.close()


def save_pr_curves(metrics, pr_curves, folder):
    indexed = metrics.set_index(["class", "iouThr", "area", "maxDet"])
    fmt = "class={class}-iouThr={iouThr}-area={area}-maxDet={maxDet}.png"
    for p in pr_curves:
        idx = p["class"], p["iouThr"], p["area"], p["maxDet"]
        recall = indexed.loc[idx, "recall"]
        precision = indexed.loc[idx, "precision"]
        plt.clf()
        plt.title("AP = {:.3f}".format(np.mean(precision)))
        plt.plot(recall, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.grid()
        plt.savefig(os.path.join(folder, fmt.format(**p)))


def score_filter(dt_json, args):
    dt_json_new = []
    for ann in dt_json:
        if ann["score"] >= args.score_thr:
            dt_json_new.append(ann)
    return dt_json_new


def main(args):
    os.makedirs(args.folder, exist_ok=True)

    coco_gt = COCO(args.gt)
    with open(args.dt, "r") as f:
        dt_json = json.load(f)
    dt_json = score_filter(dt_json, args)
    coco_dt = coco_gt.loadRes(dt_json)

    params = Params(coco_gt, iouType=args.iou_type)
    metrics = detection_metrics(coco_gt, coco_dt, params)

    save_csv(metrics, args.folder)

    save_report(metrics, args.folder)
    save_pr_curves(metrics, PR_CURVES, args.folder)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
