### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateConDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import tensorboardX
import random

GAP = 1

def write_temp(opt, target_phase):
    source_path = os.path.join(opt.dataroot, opt.phase + '_list.txt')
    target_path = os.path.join(opt.dataroot, target_phase + '_list.txt')
    f = open(source_path, 'r')
    all_paths = f.readlines()
    ans = []
    last = ''
    tot = 0
    print(len(all_paths))
    for path in all_paths:
        path_ = path.split('&')[0]
        if (path_ != last):
            last = path_
            tot = tot + 1
            if tot % GAP == 0:
                ans.append(path)
    f_w = open(target_path, 'w')
    f_w.writelines(ans)

opt = TrainOptions().parse()
opt.phase = 'val'
write_temp(opt, "temp")
opt.phase = "temp"
opt.serial_batches = True

data_loader = CreateConDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

visualizer = Visualizer(opt)

total_steps = 0# (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for i, data in enumerate(dataset):
    if (i % 100 == 0):
        print((i, dataset_size))
    visuals = OrderedDict([('input_label', util.tensor2label(data['A'][0], 256)),
                       ('real_image', util.tensor2im(data['A2'][0]))])
    visualizer.display_current_results2(visuals, 0, i)
   
