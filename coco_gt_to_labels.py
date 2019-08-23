import argparse

from pycocotools.coco import COCO

from utils import coco_cat_to_label


def build_parser():
    parser = argparse.ArgumentParser("Extract labels from COCO annotations")
    parser.add_argument("--coco-gt", type=str, required=True,
                        help="Ground truth in COCO .json format")
    parser.add_argument("--conf", action="store_true",
                        help="Add confidence attribute to labels")
    parser.add_argument("--supercat", action="store_true",
                        help="Add supercategory to label name")
    parser.add_argument("--labels-out", type=str, required=True,
                        help="Output txt file with space sep labels")
    return parser


def main(args):
    coco = COCO(args.coco_gt)
    labels = [coco_cat_to_label(cat, args.supercat) for cat in coco.cats.values()]
    labels = sorted(set(labels))
    if args.conf:
        line = " ".join(map(lambda x: x + " @text=conf:NA", labels))
    else:
        line = " ".join(labels)
    with open(args.labels_out, "w") as f:
        f.write(line)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
