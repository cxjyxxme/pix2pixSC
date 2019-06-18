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
FPS = 30

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
            if (abs(points[8, 0] - points[1, 0]) == 0 or abs(points[8, 0] - points[15, 0]) == 0):
                detected = False
            else:
                r = float(abs(points[8, 0] - points[1, 0])) / abs(points[8, 0] - points[15, 0])
                r = min(r, 1 / r)
                if (r < side_face_threshold):
                    detected = False

    return detected, points, [left_up_x, left_up_y, right_down_x, right_down_y]

def get_img(path, target_path):
    #print("233")
    img = io.imread(path)
    detected, _, box = get_keys(img)
    if (detected):
        h, w, c = img.shape
        b_h = box[3] - box[1]
        b_w = box[2] - box[0]
        dh = int((b_h / face_rate - b_h) / 2)
        dw = int((b_w / face_rate - b_w) / 2)
        ddh = int(b_h * up_center_rate)
        if (box[1] - dh - ddh < 0 or box[3] + dh - ddh >= h or box[0] - dw < 0 or box[2] + dw >= w): 
            return False, None
        else:
            img_ = img[box[1] - dh - ddh : box[3] + dh - ddh, box[0] - dw : box[2] + dw, :].copy()
        if (img_.shape[0] < out_size / 4 or img_.shape[1] < out_size / 4):
            return False, None
        #dh = int(h * (1 - crop_rate) / 2)
        #dw = int(w * (1 - crop_rate) / 2)
        #img_ = img[dh:h-dh, dw:w-dw, :].copy()
        img_ = transform.resize(img_, (out_size, out_size))
        #time_s = time.time()
        #io.imsave(target_path, img_)
        #print("writetime:" + str(time.time() - time_s))
        img_ = (np.maximum(np.minimum(255, img_ * 256), 0)).astype(np.uint8)

        return True, img_
    else:
        return False, None

def deal(datas, rootpath):
    paths = []
    path = []
    last = ''
    min_hw = [1000, 1000]
    for data in datas:
        paras = data.split(',')
        names = paras[0].split('\\')
        if (last != names[0] + '\\' + names[1]):
            if min_hw[0] > threshold and min_hw[1] > threshold:
                paths.append(path)
            path = []
            min_hw = [1000, 1000]
            last = names[0] + '\\' + names[1]
        min_hw[0] = min(min_hw[0], int(paras[4]))
        min_hw[1] = min(min_hw[1], int(paras[5]))
        path.append(os.path.join(rootpath, 'aligned_images_DB', names[0], names[1], 'aligned_detect_' + names[2]))
    
    if min_hw[0] > threshold and min_hw[1] > threshold:
        paths.append(path)
    return paths
            
def get_num(path):
    t = path.find("a.")
    return int(path[t + 2 : -4])

def deal_vid(paths):
    paths = sorted(paths, key=lambda path: get_num(path))
    ans = []
    for i in range(len(paths)):
        if (i % 10 == 0):
            ans.append(paths[i])
    return ans

def get_paths(rootpath):
    data_path = os.path.join(rootpath, 'frame_images_DB')
    data_path = rootpath
    paths = []
    for root, _, fnames in os.walk(data_path):
        temp = []
        for fname in fnames:
            if not fname.endswith('.jpg'):
                continue
            temp.append(os.path.join(root, fname))
        temp = deal_vid(temp)
        print(len(temp))
        paths.append([temp])
    random.shuffle(paths)
    paths_ = []
    for human in paths:
        for vid in human:
            paths_.append(vid)
    return paths_

def transform_name(name, root):
    name = name[len(root):]
    name = name.replace('/', '__').replace(' ', '_')
    return name

def neighbor_index(index, length, n_size):
    left = max(0, index[1] - n_size)
    right = min(length - 1, index[1] + n_size)
    return [index[0], np.random.randint(left, right + 1)]

def make_list(paths, phase, opt):
    indexs = []
    for i in range(len(paths)):
        for j in range(paths[i]['len']):
            indexs.append([i, j])

    ans_list = []
    for i in range(len(paths)):
        for j in range(paths[i]['len']):
            for k in range(opt.A_repeat_num):
                '''
                if (phase == 'val' or phase == 'test'):
                    index = indexs[np.random.randint(len(indexs))]
                    context = paths[i]['label_names'][j] \
                        + '&' + paths[index[0]]['label_names'][index[1]] + '&' + paths[index[0]]['img_names'][index[1]] \
                        + '&' + paths[i]['img_names'][j] \
                        + '\n'
                    ans_list.append(context)
                else:
                '''
                if (random.random() < opt.same_style_rate):
                    index_B = neighbor_index([i, j], paths[i]['len'], opt.neighbor_size)
                else:
                    index_B = indexs[np.random.randint(len(indexs))]
                index_C = neighbor_index(index_B, paths[index_B[0]]['len'], opt.neighbor_size)
                index_D = index_C
                while (index_D[0] == index_C[0]):
                    index_D = indexs[np.random.randint(len(indexs))]
                context = paths[i]['label_names'][j] \
                    + '&' + paths[index_B[0]]['label_names'][index_B[1]] + '&' + paths[index_B[0]]['img_names'][index_B[1]] \
                    + '&' + paths[i]['img_names'][j] \
                    + '&' + paths[index_C[0]]['label_names'][index_C[1]] + '&' + paths[index_C[0]]['img_names'][index_C[1]] \
                    + '&' + paths[index_D[0]]['label_names'][index_D[1]] + '&' + paths[index_D[0]]['img_names'][index_D[1]] \
                    + '\n'
                ans_list.append(context)

    f = open(os.path.join(opt.target_path, phase + '_list.txt'), 'w')
    f.writelines(ans_list)
    
opt = GenerateDataOptions().parse(save=False)
opt.neighbor_size *= FPS
paths = get_paths(opt.source_path)
label_path = os.path.join(opt.target_path, 'keypoints/')
img_path = os.path.join(opt.target_path, 'img/')
if (not os.path.exists(label_path)):
    os.makedirs(label_path)
if (not os.path.exists(img_path)):
    os.makedirs(img_path)
paths_ = []
ans = 0
for i in range(len(paths)):
    ans += len(paths[i])
print(ans)
tot = 0
for i in range(len(paths)):
    if (opt.copy_data):
        print(str(tot) + ' ' + str(len(paths[i])))
    img_names = []
    label_names = []
    for j in range(len(paths[i])):
        img_name = transform_name(paths[i][j], opt.source_path)
        label_name = img_name[: -3] + 'txt'
        if (opt.copy_data):
            useable, img = get_img(paths[i][j], 'datasets/YouTubeFaces/temp.jpg')
            if not useable:
                continue
            #img = io.imread('datasets/YouTubeFaces/temp.jpg')
            detected, keys, box = get_keys(img)
            if detected:
                tot += 1
                io.imsave(os.path.join(img_path, img_name), img)
                np.savetxt(os.path.join(label_path, label_name), keys, fmt='%d', delimiter=',')
                img_names.append(img_path + img_name)
                label_names.append(label_path + label_name)
        else:
            if os.path.exists(label_path + label_name):
                img_names.append(img_path + img_name)
                label_names.append(label_path + label_name)
    paths_.append({'label_names': label_names, 'img_names':img_names, 'len': len(img_names)})
paths = paths_
img_num = 0
for i in range(len(paths)):
    img_num += len(paths[i]['img_names'])
print(img_num)
val_size = int(img_num * opt.val_ratio)
test_size = int(img_num * opt.test_ratio)
did_val = False
tot = 0
last = 0
last_name = ''
for i in range(len(paths)):
    tot += len(paths[i]['img_names'])
    if (len(paths[i]['img_names']) > 0):
        name = paths[i]['img_names'][0].split('__')[1]
        if ((not did_val) and tot >= val_size and last_name != name):
            print("=============")
            did_val = True
            tot = 0
            last = i + 1
            make_list(paths[0:i], 'val', opt)
        elif (did_val and tot >= test_size and last_name != name):
            print("=============")
            make_list(paths[last:i], 'test', opt)
            last = i + 1
            break
        print(name)
        last_name = name
make_list(paths[last:], 'train', opt)

