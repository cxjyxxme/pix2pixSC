### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.generate_data_options import GenerateDataOptions
from data.data_loader import CreateDataLoader
from PIL import Image
import copy
import numpy as np
from shutil import copyfile
import random
import glob
from skimage import transform,io
import dlib
import sys
import time
import math

threshold = 130
crop_rate = 0.6
face_rate = 0.5
up_center_rate = 0.2
out_size = 512
side_face_threshold = 0.7
predictor_path = os.path.join('./datasets/', 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
time_last = time.time()

pts_max_score = 1000#.1 * 256
pic_max_score = 10#0.007 * 256
#pic_size = 256.0
pic_size = 1.0

def get_keys(img, get_point = True):
    global time_last
    time_start = time.time()
    #print("other:" + str(time_start - time_last))
    dets = detector(img, 1)
    #print(time.time() - time_start)
    time_last = time.time()
    detected = False
    points = []
    left_up_x = 10000000
    left_up_y = 10000000
    right_down_x = -1
    right_down_y = -1
        
    if len(dets) > 0:
        detected = True
        if (get_point == False):
            return True, [], [dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()]
        else:
            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y
                left_up_x = min(left_up_x, points[b, 0])
                left_up_y = min(left_up_y, points[b, 1])
                right_down_x = max(right_down_x, points[b, 0])
                right_down_y = max(right_down_y, points[b, 1])
            '''
            if (abs(points[8, 0] - points[1, 0]) == 0 or abs(points[8, 0] - points[15, 0]) == 0):
                detected = False
            else:
                r = float(abs(points[8, 0] - points[1, 0])) / abs(points[8, 0] - points[15, 0])
                r = min(r, 1 / r)
                if (r < side_face_threshold):
                    detected = False
            '''

    return detected, points, [left_up_x, left_up_y, right_down_x, right_down_y]
def transfer_list(list1):
    list2 = []
    for v in list1:
        list2.append(v[:-len('synthesized_image.jpg')] + 'real_image.jpg')
        #list2.append(v[:-len('synthesized_image.jpg')] + 'B2.jpg')
    return list2
'''

def transfer_list(list1):
    list2 = []
    for v in list1:
        id=v.split('_')[-1][:-4]
        t = 'results/label2city_256p_face_102/test_final_latest_True/images/'+id+'_real_image.jpg'
        list2.append(t)
    return list2
'''

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

path = 'results/label2city_256p_face_104/test_final_latest_True/images'

list1 = []
for root, _, fnames in os.walk(path):
    for fname in fnames:
        if (fname.endswith('_synthesized_image.jpg')):
            list1.append(os.path.join(root, fname))
list2 = transfer_list(list1)
score = 0
print(len(list1))
for i in range(len(list1)):
    img1 = io.imread(list1[i])
    img2 = io.imread(list2[i])
    detected1, points1, _ = get_keys(img1)
    detected2, points2, _ = get_keys(img2)
    if (not detected1) or (not detected2):
        temp_score = pic_max_score
    else:
        points1 = points1 / pic_size
        points2 = points2 / pic_size
        temp_score = 0
        for j in range(68):
            d = dist(points1[j, :], points2[j, :])
            d = min(pts_max_score, d)
            temp_score += d
        temp_score = min(pic_max_score, temp_score / 68)
    score += temp_score
    if (i % 10 == 0):
        print(i)
        print(score / (i + 1)/256)
        print(temp_score/256)
score /= len(list1)
print(score/256)
    

