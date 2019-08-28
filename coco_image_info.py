import argparse
import os
import json

from PIL import Image


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Look for images starting from this folder")
    parser.add_argument("--coco-gt", type=str, required=True,
                        help="Coco file with categories")
    parser.add_argument("--out", type=str, default=None,
                        help="Coco image info")
    parser.add_argument("--logfile", type=str, default="open_errors.log",
                        help="Save files that cannot be opened as image")
    parser.add_argument("--fmt", action="store_true",
                        help="Easy readable formatting")
    return parser


def main(args):
    if args.out is None:
        if args.root.endswith("/"):
            args.root = args.root[:-1]
        args.out = args.root + ".json"
    images = []
    image_id = 0
    with open(args.logfile, "w+") as f:
        for path, dirs, files in os.walk(args.root):
            for name in files:
                full_name = os.path.join(path, name)
                try:
                    img = Image.open(full_name)
                    width, height = img.size
                    image = dict(
                        id=image_id,
                        file_name=os.path.relpath(full_name, args.root),
                        height=height,
                        width=width
                    )
                    images.append(image)
                    image_id += 1
                except IOError:
                    print(full_name, file=f)
    with open(args.coco_gt, "r") as f:
        categories = json.load(f)["categories"]
    coco = {"images": images, "categories": categories}
    indent = None
    separators = (",", ":")
    if args.fmt:
        indent = 2
        separators = (",", ": ")
    with open(args.out, "w") as f:
        json.dump(coco, f, indent=indent, separators=separators)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
