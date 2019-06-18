### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateConDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import random
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateConDataLoader(opt)
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
    
tot = 0
for i, data in enumerate(dataset, start=random.randint(0, len(dataset) - opt.how_many)):
    tot += 1
    if tot > opt.how_many:
        break
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
