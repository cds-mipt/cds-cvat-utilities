import argparse
import numpy as np

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

        # пороги IoU
        self.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        # площади объектов, для которых будут вычислeны метрики
        self.areas = {
            "small": [0 ** 2, 32 ** 2],
            "medium": [32 ** 2, 96 ** 2],
            "large": [96 ** 2, 1e5 ** 2],
            "all": [0 ** 2, 1e5 ** 2]
        }

        # остальное, как правило, нет причин менять
        self.id_to_class = {cat_id: cat["name"] for cat_id, cat in gt.cats.items()}
        self.id_to_class[-1] = "all"
        self.catIds = list(gt.cats.keys())
        self.useCats = 1
        self.iouType = iouType
        self.useSegm = None
        self.maxDets = [300]
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.areaRngLbl = list(self.areas.keys())
        self.areaRng = [self.areas[k] for k in self.areaRngLbl]


def build_parser():
    parser = argparse.ArgumentParser("Instance segmentation and object detection metrics")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--dt", type=str, required=True)
    parser.add_argument("--iou-type", type=str, default="segm", choices=["bbox", "segm"])
    parser.add_argument("--result", type=str, default="result.txt")
    return parser


def summarize(gt, coco_eval):
    def calk_cond_mean(s, area, cat_id=-1, iouThr="mean"):
        p = coco_eval.params
        s = s[:, :, list(p.areaRngLbl).index(area), -1]
        if cat_id != -1:
            s = s[:, p.catIds.index(cat_id)]
        if iouThr != "mean":
            s = s[list(p.iouThrs).index(iouThr)]
        valid = s > -1
        return np.mean(s[valid]) if valid.any() else -1

    def AP(area, cat_id=-1, iouThr=None):
        s = coco_eval.eval['precision'].mean(axis=1)
        return calk_cond_mean(s, area, cat_id, iouThr)

    def AR(area, cat_id=-1, iouThr=None):
        s = coco_eval.eval['recall']
        return calk_cond_mean(s, area, cat_id, iouThr)

    msg = "[class = {:>10s} | area = {:6s} | IoU = {:<9}]\tAP = {:0.3f}\tAR = {:0.3f}"

    stats = []
    p = coco_eval.params
    for cat_id in [-1] + p.catIds:
        for area in p.areaRngLbl:
            for iouThr in ["mean"] + list(p.iouThrs):
                ap = AP(area, cat_id, iouThr)
                ar = AR(area, cat_id, iouThr)
                print(msg.format(p.id_to_class[cat_id], area, iouThr, ap, ar))

    coco_eval.stats = np.array(stats)


def main(args):
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

    summarize(coco_gt, coco_eval)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
