### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        data_list_path = os.path.join(opt.dataroot, opt.phase + '_list.txt')
        f = open(data_list_path, 'r')
        self.all_paths = f.readlines()
        self.dataset_size = len(self.all_paths) 
      
    def transfer_path(self, path):
        temp = path.split('__')[-1]
        temp = './datasets/bdd2/danet_vis/' + temp + '.label.png'
        return temp

    def same_style(self, path1, path2):
        t1 = path1.split('__')
        t2 = path2.split('__')
        p1 = t1[-3] + '_' + t1[-2]
        p2 = t2[-3] + '_' + t2[-2]
        if (p1 == p2):
            return 1
        else:
            return 0

    def get_X(self, path, params, do_transfer = False):
        ### input A (label maps)
        if self.opt.use_new_label and do_transfer:
            path = self.transfer_path(path)
        A = Image.open(path)        
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
        return A_tensor

    def get_X2(self, path, params):
        B = Image.open(path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)
        return B_tensor

    def __getitem__(self, index):        
        paths = self.all_paths[index].rstrip('\n').split('&')
        A = Image.open(paths[0])        
        params = get_params(self.opt, A.size)

        A_tensor = self.get_X(paths[0], params)
        B_tensor = self.get_X(paths[1], params, True)
        B2_tensor = self.get_X2(paths[2], params)
        A2_tensor = self.get_X2(paths[3], params)
        C_tensor = C2_tensor = D_tensor = D2_tensor = 0
        if (self.opt.isTrain):
            C_tensor = self.get_X(paths[4], params)
            C2_tensor = self.get_X2(paths[5], params)
            D_tensor = self.get_X(paths[6], params)
            D2_tensor = self.get_X2(paths[7], params)
        input_dict = {'A': A_tensor, 'A2': A2_tensor, 'B': B_tensor, 'B2': B2_tensor, 'C': C_tensor, 'C2': C2_tensor,
                'D': D_tensor, 'D2': D2_tensor, 'path': paths[0] + '_' + paths[1], 'same_style': self.same_style(paths[2], paths[3])}
        return input_dict

    def __len__(self):
        return len(self.all_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset2'
