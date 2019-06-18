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

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateConDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated = model(
                Variable(data['A']), Variable(data['A2']),
                Variable(data['B']), Variable(data['B2']),
                Variable(data['C']), Variable(data['C2']),
                Variable(data['D']), Variable(data['D2']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))
        
        if (total_steps <= opt.use_style_iter):
            use_style = 0
        else:
            use_style = 1
        loss_dict['G_GAN_style'] *= 10
        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_SD = (loss_dict['SD_fake1'] + loss_dict['SD_fake2'] + loss_dict['SD_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_style'] * use_style + (loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0))

        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        # update discriminator weights
        if (total_steps > opt.use_style_iter):
            model.module.optimizer_SD.zero_grad()
            loss_SD.backward()
            model.module.optimizer_SD.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors['loss_G'] = loss_G
            errors['loss_D'] = loss_D
            errors['loss_SD'] = loss_SD
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)


        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['A'][0], 256)),
                                   ('real_image', util.tensor2im(data['A2'][0])),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('B', util.tensor2label(data['B'][0], 256)),
                                   ('B2', util.tensor2im(data['B2'][0]))])
            visualizer.display_current_results2(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if opt.use_iter_decay and total_steps > opt.niter_iter:
            for temp in range(opt.batchSize):
                model.module.update_learning_rate()

        if opt.use_iter_decay and total_steps >= opt.niter_iter + opt.niter_decay_iter:
            break
        if epoch_iter >= dataset_size:
            break
 
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter and not opt.use_iter_decay:
        model.module.update_learning_rate()
        
    if opt.use_iter_decay and total_steps > opt.niter_iter + opt.niter_decay_iter:
            break      
    
