import cv2              # pip install opencv-python
import os
from os.path import join
import argparse
import random
import numpy as np

in_path = "../FaceForensics/datas/"
out_path = "../FaceForensics/out_data/"

def process(vid_path, out_path):
    print(vid_path)
    print(out_path)
    os.makedirs(out_path)
    video = cv2.VideoCapture(vid_path)
    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(n):
        _, image = video.read()
        path = os.path.join(out_path, 'a.' + str(i) + '.jpg')
        cv2.imwrite(path, image)

tot = 0
for root, _, fnames in os.walk(in_path):
    for fname in fnames:
        if not fname.endswith('.avi'):
            continue
        process(os.path.join(root, fname), os.path.join(out_path, "Mr_"+str(tot), "1"))
        tot += 1
