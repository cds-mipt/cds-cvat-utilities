import argparse
import json

from collections import defaultdict


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-in", type=str, required=True)
    parser.add_argument("--md-image", type=int, default=None)
    parser.add_argument("--md-class", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--coco-out", type=str, required=True)
    return parser


def max_dets_per_class(grouped, t):
    for image_id, image in grouped.items():
        for category_id, anns in image.items():
            grouped[image_id][category_id] = anns[:t]
    return grouped


def max_dets_per_image(grouped, t):
    for image_id, image in grouped.items():
        anns_all = []
        for category_id, anns in image.items():
            anns_all.extend(anns)
        anns_all = sorted(anns_all, key=lambda x: x["score"], reverse=True)
        anns_all = anns_all[:t]
        grouped[image_id] = defaultdict(list)
        for ann in anns_all:
            grouped[image_id][ann["category_id"]].append(ann)
    return grouped


def score_threshold(grouped, t):
    for image_id, image in grouped.items():
        for category_id, anns in image.items():
            anns_new = []
            for ann in anns:
                if ann["score"] >= t:
                    anns_new.append(ann)
            grouped[image_id][category_id] = anns_new
    return grouped


def group(coco_dt):
    grouped = defaultdict(lambda: defaultdict(list))
    for ann in coco_dt:
        grouped[ann["image_id"]][ann["category_id"]].append(ann)
    for image_id, image in grouped.items():
        for category_id, anns in image.items():
            grouped[image_id][category_id] = sorted(anns, key=lambda x: x["score"], reverse=True)
    return grouped


def ungroup(grouped):
    coco_dt = []
    for image_id, image in grouped.items():
        for category_id, anns in image.items():
            for ann in anns:
                coco_dt.append(ann)
    return coco_dt


def main(args):
    with open(args.coco_in, "r") as f:
        coco_dt = json.load(f)

    grouped = group(coco_dt)

    if args.md_class:
        grouped = max_dets_per_class(grouped, args.md_class)
    if args.md_image:
        grouped = max_dets_per_image(grouped, args.md_image)
    if args.score_thr:
        grouped = score_threshold(grouped, args.score_thr)

    coco_dt = ungroup(grouped)

    with open(args.coco_out, "w") as f:
        json.dump(coco_dt, f, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
