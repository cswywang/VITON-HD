import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pth", type=str, default="0")
args = parser.parse_args()

img=cv2.imread(args.pth)
img = cv2.resize(src=img, dsize=(768, 1024), fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
cv2.imwrite(args.pth, img)