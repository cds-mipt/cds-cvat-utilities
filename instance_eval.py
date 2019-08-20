import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


np.warnings.filterwarnings("ignore")


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

        self.plot_PR = [
            {"iou": 0.5, "area": "all", "class": "person", "maxDets": 10},
            {"iou": 0.75, "area": "all", "class": "person", "maxDets": 10}
        ]

        # остальное, как правило, нет причин менять
        self.id_to_class = {cat_id: cat["name"] for cat_id, cat in gt.cats.items()}
        self.id_to_class[-1] = "all"
        self.class_to_id = {cat["name"]: cat_id for cat_id, cat in gt.cats.items()}
        self.catIds = [self.class_to_id[cls] for cls in self.classes] or list(gt.cats.keys())
        self.useCats = 1
        self.iouType = iouType
        self.useSegm = None
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.areaRngLbl = list(self.areas.keys())
        self.areaRng = [self.areas[k] for k in self.areaRngLbl]


def build_parser():
    parser = argparse.ArgumentParser("Instance segmentation and object detection metrics")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--dt", type=str, required=True)
    parser.add_argument("--iou-type", type=str, default="segm", choices=["bbox", "segm"])
    parser.add_argument("--folder", type=str, default="results")
    return parser


def summarize(coco_eval, f):
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

    mean_msg = "[area = {:6s} | IoU = {:<4} | maxDets = {:<3} ]  mAP = {:0.3f}  mAR = {:0.3f}"
    class_msg = "[class = {:>20s} | area = {:>6s} | IoU = {:<4} | maxDets = {:<3} ]  AP = {:0.3f}  AR = {:0.3f}"

    stats = []
    p = coco_eval.params
    for area in p.areaRngLbl:
        for maxDet in p.maxDets:
            for iouThr in ["mean"] + list(p.iouThrs):
                ap = AP(area, -1, iouThr, maxDet)
                ar = AR(area, -1, iouThr, maxDet)
                print(mean_msg.format(area, iouThr, maxDet, ap, ar), file=f)
            print(file=f)
    print(file=f)
    for cat_id in p.catIds:
        for area in p.areaRngLbl:
            for maxDet in p.maxDets:
                for iouThr in ["mean"] + list(p.iouThrs):
                    ap = AP(area, cat_id, iouThr, maxDet)
                    ar = AR(area, cat_id, iouThr, maxDet)
                    print(class_msg.format(p.id_to_class[cat_id], area, iouThr, maxDet, ap, ar), file=f)
                print(file=f)

    coco_eval.stats = np.array(stats)


def plot_PR_curve(coco_eval, folder):
    p = coco_eval.params
    recall = p.recThrs
    fmt = "class={class}-iou={iou}-area={area}-maxDets={maxDets}.png"
    for params in p.plot_PR:
        ti = list(p.iouThrs).index(params["iou"])
        ki = list(p.catIds).index(p.class_to_id[params["class"]])
        ai = list(p.areaRngLbl).index(params["area"])
        di = list(p.maxDets).index(params["maxDets"])
        precision = coco_eval.eval['precision'][ti, :, ki, ai, di]
        plt.clf()
        plt.title("AP = {:.3f}".format(precision.mean()))
        plt.plot(recall, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.grid()
        plt.savefig(os.path.join(folder, fmt.format(**params)))


def main(args):
    os.makedirs(args.folder, exist_ok=True)

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.dt)

    params = Params(coco_gt, iouType=args.iou_type)
    if not params.imgIds:
        params.imgIds = sorted(coco_gt.getImgIds())
    if not params.catIds:
        params.catIds = sorted(coco_gt.getCatIds())

    coco_eval = COCOeval(coco_gt, coco_dt, args.iou_type)
    coco_eval.params = params
    coco_eval.evaluate()
    coco_eval.accumulate()

    with open(os.path.join(args.folder, "metrics.txt"), "w") as f:
        summarize(coco_eval, f)
    plot_PR_curve(coco_eval, args.folder)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
