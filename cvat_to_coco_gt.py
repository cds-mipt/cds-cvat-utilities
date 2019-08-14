import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat", type=str, required=True)
    parser.add_argument("--coco-in", type=str, required=False, default=None)
    parser.add_argument("--coco-out", type=str, required=True)
    return parser


def main(args):
    raise NotImplementedError


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
