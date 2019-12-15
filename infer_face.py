### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateFaceConDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import random
import torch
from skimage import transform,io
import dlib
import shutil
import numpy as np

predictor_path = os.path.join('./datasets/', 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_keys(img, get_point = True):
    dets = detector(img, 1)
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

    return detected, points, [left_up_x, left_up_y, right_down_x, right_down_y]

def get_img(path):
    face_rate = 0.5
    up_center_rate = 0.2
    out_size = 512
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
        img_ = transform.resize(img_, (out_size, out_size))
        img_ = (np.maximum(np.minimum(255, img_ * 256), 0)).astype(np.uint8)
        return True, img_
    else:
        return False, None

def get_info(path, img_path, key_path):
    useable, img = get_img(path)
    assert useable
    detected, keys, box = get_keys(img)
    assert detected
    io.imsave(img_path, img)
    np.savetxt(key_path, keys, fmt='%d', delimiter=',')
    return img_path, key_path


def make_data():
    assert os.path.exists('inference/infer_list.txt')
    with open('inference/infer_list.txt', 'r') as f:
        tasks = f.readlines()
    if os.path.exists('inference/data'):
        shutil.rmtree('inference/data')
    if os.path.exists('inference/output'):
        shutil.rmtree('inference/output')
    os.makedirs('inference/data')
    os.makedirs('inference/output')
    img_path = 'inference/data/img'
    key_path = 'inference/data/keypoints'
    os.makedirs(img_path)
    os.makedirs(key_path)

    ans_list = []
    for i in range(len(tasks)):
        name = '__Mr_'+str(i)+'__0__a.'
        img1, key1 = get_info(tasks[i].split(' ')[0], os.path.join(img_path, name + '0.jpg'), os.path.join(key_path, name + '0.txt'))
        img2, key2 = get_info(tasks[i].split(' ')[1].rstrip('\n'), os.path.join(img_path, name + '1.jpg'), os.path.join(key_path, name + '1.txt'))

        context = key1 \
            + '&' + key2 + '&' + img2 \
            + '&' + img1 \
            + '&' + key1 + '&' + img1 \
            + '&' + key1 + '&' + img1 \
            + '\n'
        ans_list.append(context)
    f = open('inference/data/infer_list.txt', 'w')
    f.writelines(ans_list)

make_data()
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateFaceConDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.which_epoch, str(opt.serial_batches)))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset, start=0):
    if opt.data_type == 16:
        data['A'] = data['A'].half()
        data['A2'] = data['A2'].half()
        data['B'] = data['B'].half()
        data['B2'] = data['B2'].half()
    elif opt.data_type == 8:
        data['A'] = data['A'].uint8()
        data['A2'] = data['A2'].uint8()
        data['B'] = data['B'].uint8()
        data['B2'] = data['B2'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        generated = model.inference(data['A'], data['B'], data['B2'])
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['A'][0], 0)),
                           ('real_image', util.tensor2im(data['A2'][0])),
                           ('synthesized_image', util.tensor2im(generated.data[0])),
                           ('B', util.tensor2label(data['B'][0], 0)),
                           ('B2', util.tensor2im(data['B2'][0]))])
    img_path = data['path']
    img_path[0] = str(i)
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
