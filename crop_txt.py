import argparse
import os

from tqdm import tqdm

def build_parser():
    parser = argparse.ArgumentParser("Add polygons according to sign class")
    parser.add_argument(
        "--input-folder",
        type=str
    )
    parser.add_argument(
        "--output-folder",
        type=str
    )
    parser.add_argument(
        "--min-height",
        type=str
    )
    parser.add_argument(
        "--min-width",
        type=str
    )
    return parser


def check(line):
    line = line.split()
    xtl, ytl, xbr, ybr = line[1:]
    if int(xbr) - int(xtl) < int(args.min_width):
        return 0
    elif int(ybr) - int(ytl) < int(args.min_height):
        return 0
    else:
        return 1

def crop_file(file):
    lines = file.readlines()
    str = ""
    for line in lines:
        if check(line) == 0:
            lines.remove(line)
    for line in lines:
        str += line
    return str




def main(args):

    for file in os.listdir(args.output_folder):
        file_path = os.path.join(args.output_folder, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    for file in tqdm(os.listdir(args.input_folder)):
        f_in = open(args.input_folder+"/" + file, 'r')
        f_out = open(args.output_folder+"/" + file, 'w')
        str = crop_file(f_in)
        f_out.write(str)
        f_in.close()
        f_out.close()

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
