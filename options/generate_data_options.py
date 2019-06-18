### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class GenerateDataOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--neighbor_size', type=int, default=int(30), help='max distance between two same style frame.')      
        self.parser.add_argument('--same_style_rate', type=float, default=float(0.5), help='rate of A&B are neighbors')      
        self.parser.add_argument('--copy_data', action='store_true', default=False, help='copy img datas')      
        self.parser.add_argument('--target_path', type=str, default='./datasets/apollo/', help='target path')      
        self.parser.add_argument('--source_path', type=str, default='../datas/road02/', help='target path')      
        self.parser.add_argument('--val_ratio', type=float, default=float(0.2), help='val data ratio')      
        self.parser.add_argument('--test_ratio', type=float, default=float(0.2), help='test data ratio')      
        self.parser.add_argument('--A_repeat_num', type=int, default=int(10), help='# of same A repeats in dataset.')      
        self.isTrain = False
