### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from PIL import Image
import torch
from data.base_dataset import get_params, get_transform, normalize
import copy
from models import networks
import numpy as np
import torch.nn as nn

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

def get_features(inst, feat):
    feat_num = opt.feat_num
    h, w = inst.size()[1], inst.size()[2]
    block_num = 32
    feature = {}
    max_v = {}
    for i in range(opt.label_nc):
        feature[i] = np.zeros((0, feat_num+1))
        max_v[i] = 0
    for i in np.unique(inst):
        label = i if i < 1000 else i//1000
        idx = (inst == int(i)).nonzero()
        num = idx.size()[0]
        idx = idx[num//2,:]
        val = np.zeros((feat_num))                        
        for k in range(feat_num):
            val[k] = feat[0, idx[0] + k, idx[1], idx[2]].data[0]    
        temp = float(num) / (h * w // block_num)
        if (temp > max_v[label]):
            max_v[label] = temp
            feature[label] = val
    return feature

def getitem(A_path, B_path, inst_path, feat_path):        
    ### input A (label maps)
    A = Image.open(A_path)        
    params = get_params(opt, A.size)
    if opt.label_nc == 0:
        transform_A = get_transform(opt, params)
        A_tensor = transform_A(A.convert('RGB'))
    else:
        transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(A) * 255.0

    B_tensor = inst_tensor = feat_tensor = 0
    ### input B (real images)
    B = Image.open(B_path).convert('RGB')
    transform_B = get_transform(opt, params)      
    B_tensor = transform_B(B)

    ### if using instance maps        
    inst = Image.open(inst_path)
    inst_tensor = transform_A(inst)

    #get feat
    netE  = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=opt.gpu_ids)  
    feat_map = netE.forward(Variable(B_tensor[np.newaxis, :].cuda(), volatile=True), inst_tensor[np.newaxis, :].cuda())
    '''
    feat_map = nn.Upsample(scale_factor=2, mode='nearest')(feat_map)
    image_numpy = util.tensor2im(feat_map.data[0])
    util.save_image(image_numpy, feat_path)

    feat = Image.open(feat_path).convert('RGB')
    norm = normalize()
    feat_tensor = norm(transform_A(feat))                            
    input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
            'feat': feat_tensor, 'path': A_path}
    '''

    return get_features(inst_tensor, feat_map)

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

label_path = 'datasets/test/label.png'
img_path = 'datasets/test/img.png'
inst_path = 'datasets/test/inst.png'
feat_path = 'datasets/test/feat.png'
con = getitem(label_path, img_path, inst_path, feat_path)
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
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
    elif opt.conditioned:
        generated = model.inference_conditioned(data['label'], data['inst'], con)
    else:
        generated = model.inference(data['label'], data['inst'])
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
