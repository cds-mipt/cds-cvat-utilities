import argparse
import os
import numpy as np

from tempfile import NamedTemporaryFile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from markup import VisionMarkup

np.warnings.filterwarnings("ignore")


class Params:

    def __init__(self, gt, iouType):
        """
        iouType - one of 'bbox', 'segm'
        """
        # список id изображений для подсчета метрик
        # пустой - использовать все
        self.imgIds = []

        # список id классов для подсчета метрик
        # пустой - использовать все
        self.classes = ["bus", "car"]

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
        self.catIds = [gt.object_label_ids[cls] for cls in self.classes]
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
    parser.add_argument("--gt-format", type=str, default="cvat")
    parser.add_argument("--rt", type=str, required=True)
    parser.add_argument("--rt-format", type=str, default="cvat")
    parser.add_argument("--iou-type", type=str, default="segm", choices=["bbox", "segm"])
    parser.add_argument("--result", type=str, default="result.txt")
    return parser


def summarize(gt, coco_eval):
    def calk_cond_mean(s, area, cls="all", iouThr="mean"):
        p = coco_eval.params
        s = s[:, :, list(p.areaRngLbl).index(area), -1]
        if cls != "all":
            s = s[:, p.catIds.index(gt.object_label_ids[cls])]
        if iouThr != "mean":
            s = s[list(p.iouThrs).index(iouThr)]
        valid = s > -1
        return np.mean(s[valid]) if valid.any() else -1

    def AP(area, cls="all", iouThr=None):
        s = coco_eval.eval['precision'].mean(axis=1)
        return calk_cond_mean(s, area, cls, iouThr)

    def AR(area, cls="all", iouThr=None):
        s = coco_eval.eval['recall']
        return calk_cond_mean(s, area, cls, iouThr)

    msg = "[class = {:>10s} | area = {:6s} | IoU = {:<9}]\tAP = {:0.3f}\tAR = {:0.3f}"

    stats = []
    p = coco_eval.params
    for cls in ["all"] + p.classes:
        for area in p.areaRngLbl:
            for iouThr in ["mean"] + list(p.iouThrs):
                ap = AP(area, cls, iouThr)
                ar = AR(area, cls, iouThr)
                print(msg.format(cls, area, iouThr, ap, ar))

    coco_eval.stats = np.array(stats)


def main(args):
    coco_gt_file = NamedTemporaryFile("w", delete=False)
    coco_gt_file.close()
    gt = VisionMarkup()
    gt.load_cvat(args.gt)

    coco_rt_file = NamedTemporaryFile("w", delete=False)
    coco_rt_file.close()
    rt = VisionMarkup()
    rt.load_cvat(args.rt)

    # в фаиле с результатом классы могли быть перечислины в другом порядке или отсутствовать
    rt.object_label_ids = gt.object_label_ids

    gt.dump_coco(coco_gt_file.name, "gt")
    rt.dump_coco(coco_rt_file.name, "rt")

    coco_gt = COCO(coco_gt_file.name)
    coco_rt = coco_gt.loadRes(coco_rt_file.name)

    params = Params(gt=gt, iouType=args.iou_type)
    if not params.imgIds:
        params.imgIds = sorted(coco_gt.getImgIds())
    if not params.catIds:
        params.catIds = sorted(coco_gt.getCatIds())

    coco_eval = COCOeval(coco_gt, coco_rt, args.iou_type)
    coco_eval.params = params
    coco_eval.evaluate()
    coco_eval.accumulate()

    summarize(gt, coco_eval)

    os.remove(coco_gt_file.name)
    os.remove(coco_rt_file.name)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
