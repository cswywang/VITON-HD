import human_part_segmentation_atr
import human_part_segmentation_lip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--person", type=str, default="0")
args = parser.parse_args()

human_part_segmentation_atr.atr(args.person)
human_part_segmentation_lip.atr(args.person)