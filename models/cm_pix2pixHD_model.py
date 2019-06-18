### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class CMPix2PixHDModel(BaseModel):
    def name(self):
        return 'CMPix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, True, True, True, True, True, True, use_gan_feat_loss, use_vgg_loss, True, True, True, use_gan_feat_loss, use_vgg_loss)
        def loss_filter(g_gan, d_real, d_fake, sd_fake1, sd_fake2, sd_real, g_gan_style, g_gan_feat, g_vgg, g_confidence, d_fake_final, fg_gan, fg_gan_feat, fg_vgg):
            return [l for (l,f) in zip((g_gan, d_real, d_fake, sd_fake1, sd_fake2, sd_real, g_gan_style, g_gan_feat, g_vgg, g_confidence, d_fake_final, fg_gan, fg_gan_feat, fg_vgg),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc * 2 + opt.output_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc + 1, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)    
        # Fine Generator network
        netFG_input_nc = input_nc + opt.output_nc + 1        
        self.netFG = networks.define_G(netFG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Style Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netSD_input_nc = (input_nc + opt.output_nc) * 2
            if not opt.no_instance:
                netSD_input_nc += 2
            self.netSD = networks.define_D(netSD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            self.load_network(self.netFG, 'FG', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
                self.load_network(self.netSD, 'SD', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.confidenceLoss = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','D_real', 'D_fake', 'SD_fake1', 'SD_fake2', 'SD_real', 'G_GAN_style', 'G_GAN_Feat', 'G_VGG', 
                    'G_confidence', 'D_fake_final', 'FG_GAN', 'FG_GAN_Feat', 'FG_VGG')
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))   

            # optimizer FG
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netFG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netFG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_FG = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                             

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                      

            # optimizer SD                        
            params = list(self.netSD.parameters())    
            self.optimizer_SD = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input_label(self, label_map, infer):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            if (len(self.opt.label_indexs) > 0):
                oneHot_size = (size[0], 256, size[2], size[3])
                input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
                input_label = input_label[:, self.opt.label_indexs, :, :]
            else:
                oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
                input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
        input_label = Variable(input_label, volatile=infer)
        return input_label

    def encode_input_img(self, real_image):
        return Variable(real_image.data.cuda())


    def encode_input(self, A, B, B2, A2=None, C=None, C2=None, D=None, D2=None, infer=False):             
        A_ = self.encode_input_label(A, infer)
        B_ = self.encode_input_label(B, infer)
        B2_ = self.encode_input_img(B2)
        A2_ = C_ = C2_ = D_ = D2_ = None
        if A2 is not None:
            A2_ = self.encode_input_img(A2)
        if C is not None:
            C_ = self.encode_input_label(C, infer)
        if C2 is not None:
            C2_ = self.encode_input_img(C2)
        if D is not None:
            D_ = self.encode_input_label(D, infer)
        if D2 is not None:
            D2_ = self.encode_input_img(D2)

        return A_, B_, B2_, A2_, C_, C2_, D_, D2_

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def style_discriminate(self, input_label, test_image, B, B2, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach(), B, B2), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netSD.forward(fake_query)
        else:
            return self.netSD.forward(input_concat)

    def forward(self, A_, A2_, B_, B2_, C_, C2_, D_, D2_, infer=False):
        # Encode Inputs
        A, B, B2, A2, C, C2, D, D2 = self.encode_input(A_, B_, B2_, A2_, C_, C2_, D_, D2_)  
        # forward
        fake_image_ = self.netG.forward(torch.cat((A, B, B2), dim=1))
        fake_image = fake_image_[:, 0:self.opt.output_nc, :, :]
        mask = fake_image_[:, self.opt.output_nc:self.opt.output_nc + 1, :, :]
        mask = torch.clamp(mask, 0, 1)
        details_image = self.netFG.forward(torch.cat((fake_image, mask, A), dim=1))
        final_fake_image = fake_image * mask + details_image * (1 - mask)
        '''
        STAGE 1
        '''
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(A, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(A, A2)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((A, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
       
        #style discriminator
        pred_style_fake1 = self.style_discriminate(A, fake_image, B, B2)
        loss_SD_fake1 = self.criterionGAN(pred_style_fake1, False)
        
        pred_style_real = self.style_discriminate(C, C2, B, B2)
        loss_SD_real = self.criterionGAN(pred_style_real, True)
        
        pred_style_fake2 = self.style_discriminate(D, D2, B, B2)
        loss_SD_fake2 = self.criterionGAN(pred_style_fake2, False)
        
        #GAN style loss
        pred_style_gan = self.netSD.forward(torch.cat((A, fake_image, B, B2), dim=1))        
        loss_G_GAN_style = self.criterionGAN(pred_style_gan, True)           
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, A2) * self.opt.lambda_feat
        '''
        STAGE 2
        '''
        #confidence loss
        loss_G_confidence = self.confidenceLoss(mask, torch.ones_like(mask))

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(A, final_fake_image, use_pool=True)
        loss_D_fake_final = self.criterionGAN(pred_fake_pool, False)        

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((A, final_fake_image), dim=1))        
        loss_FG_GAN = self.criterionGAN(pred_fake, True)               
       
        '''
        #style discriminator
        pred_style_fake1 = self.style_discriminate(A, final_fake_image, B, B2)
        loss_SD_fake1 = self.criterionGAN(pred_style_fake1, False)
        
        pred_style_real = self.style_discriminate(C, C2, B, B2)
        loss_SD_real = self.criterionGAN(pred_style_real, True)
        
        pred_style_fake2 = self.style_discriminate(D, D2, B, B2)
        loss_SD_fake2 = self.criterionGAN(pred_style_fake2, False)

        #GAN style loss
        pred_style_gan = self.netSD.forward(torch.cat((A, final_fake_image, B, B2), dim=1))        
        loss_FG_GAN_style = self.criterionGAN(pred_style_gan, True)           
        '''

        # GAN feature matching loss
        loss_FG_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_FG_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_FG_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_FG_VGG = self.criterionVGG(final_fake_image, A2) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter(loss_G_GAN, loss_D_real, loss_D_fake, loss_SD_fake1, loss_SD_fake2, loss_SD_real, 
            loss_G_GAN_style, loss_G_GAN_Feat, loss_G_VGG, 
            loss_G_confidence, loss_D_fake_final, loss_FG_GAN, loss_FG_GAN_Feat, loss_FG_VGG), None if not infer else fake_image, mask, final_fake_image ]

    def inference_conditioned(self, label, inst, con, k = -1):
        # Encode Inputs        
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)

        # Fake Generation
        if self.use_features:       
            # sample clusters from precomputed features             
            feat_map = self.sample_features_conditioned(inst_map, con, k)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image


    def inference(self, A_, B_, B2_):
        # Encode Inputs       
        A, B, B2, _, _, _, _, _ = self.encode_input(Variable(A_), Variable(B_), Variable(B2_), infer=True)
        '''
        # Fake Generation
        if self.use_features:       
            # sample clusters from precomputed features             
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
        input_concat = torch.cat((A, B, B2), dim=1)
        '''
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image_ = self.netG.forward(torch.cat((A, B, B2), dim=1))
                fake_image = fake_image_[:, 0:self.opt.output_nc, :, :]
                mask = fake_image_[:, self.opt.output_nc:self.opt.output_nc + 1, :, :]
                mask = torch.clamp(mask, 0, 1)
                details_image = self.netFG.forward(torch.cat((fake_image, mask, A), dim=1))
                final_fake_image = fake_image * mask + details_image * (1 - mask)
        else:
            pass
        return fake_image, mask, final_fake_image

    def sample_features_conditioned(self, inst, con, kkk = -1): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if ((con[label].size > 0) and (kkk == -1)):
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = con[label][k]
            elif label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                if (kkk != -1):
                    cluster_idx = min(kkk, feat.shape[0] - 1)
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netFG, 'FG', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netSD, 'SD', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netFG.parameters())       
        self.optimizer_FG = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        if (self.opt.use_iter_decay):
            lrd = self.opt.lr / self.opt.niter_decay_iter
        else:
            lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_SD.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_FG.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(CMPix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
