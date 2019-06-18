import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2
from skimage import feature

from data.base_dataset import BaseDataset, get_transform, get_params
from data.keypoint2img import interpPoints, drawEdge, read_keypoints2
import random

class PoseDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot                
        data_list_path = os.path.join(opt.dataroot, opt.phase + '_list.txt')
        f = open(data_list_path, 'r')
        self.all_paths = f.readlines()
        self.dataset_size = len(self.all_paths)
        if (not opt.isTrain and opt.serial_batches):
            ff = open(opt.test_delta_path, "r")
            t = int(ff.readlines()[0])
            self.delta = t
        else:
            self.delta = 0
      

    def get_image2(self, A_path, size, params, input_type):
        is_img = input_type == 'img'
        method = Image.BICUBIC if is_img else Image.NEAREST
        transform_scaleA = get_transform(self.opt, params, normalize=is_img, method=method)
        if input_type != 'openpose':
            A_img = Image.open(A_path).convert('RGB')
        else:            
            random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
            A_img, dist_tensor = read_keypoints2(A_path, size, transform_scaleA, random_drop_prob, False)  

        if input_type == 'densepose' and self.opt.isTrain:
            # randomly remove labels
            A_np = np.array(A_img)
            part_labels = A_np[:,:,2]            
            for part_id in range(1, 25):
                if (np.random.rand() < self.opt.random_drop_prob):
                    A_np[(part_labels == part_id), :] = 0
            if self.opt.remove_face_labels:            
                A_np[(part_labels == 23) | (part_labels == 24), :] = 0
            A_img = Image.fromarray(A_np)

        A_scaled = transform_scaleA(A_img)          
        if (input_type == 'openpose' and self.opt.do_pose_dist_map):
            A_scaled = torch.cat([A_scaled, dist_tensor])
        return A_scaled

    def get_X(self, path, params, B_size, B_tensor):
        paths = path.split('|')
        if not self.opt.openpose_only:
            Di = self.get_image2(paths[0], B_size, params, input_type='densepose')
            Di[2,:,:] = Di[2,:,:] * 255 / 24
        if not self.opt.densepose_only:
            Oi = self.get_image2(paths[1], B_size, params, input_type='openpose')

        if self.opt.openpose_only:
            Ai = Oi
        elif self.opt.densepose_only:
            Ai = Di
        else:
            Ai = torch.cat([Di, Oi])
        valid = (Di[0, :, :] > 0) + (Di[1, :, :] > 0) + (Di[2, :, :] > 0)
        valid_ = valid == 0
        if (self.opt.add_mask):
            B_tensor[:, valid_] = 0
        '''
        valid_ = (valid > 0)[np.newaxis, :, :]
        valid__ = torch.cat([valid_, valid_, valid_])
        print(valid__[:, 128, 128])
        B_out = B_tensor * valid__
        '''
        #Ai, Bi = self.crop(Ai), self.crop(Bi) # only crop the central half region to save time
        return Ai, B_tensor

    def same_style(self, path1, path2):
        t1 = path1.split('__')
        t2 = path2.split('__')
        p1 = t1[-3] + '_' + t1[-2]
        p2 = t2[-3] + '_' + t2[-2]
        if (p1 == p2):
            return 1
        else:
            return 0

    def get_X2(self, path, params):
        B = Image.open(path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)
        return B_tensor, B

    def __getitem__(self, index):        
        index = (index + self.delta) % len(self.all_paths)
        paths = self.all_paths[index].rstrip('\n').split('&')
        A = Image.open(paths[2])        
        params = get_params(self.opt, A.size)

        B2_tensor_, B = self.get_X2(paths[2], params)
        A2_tensor_, A = self.get_X2(paths[3], params)
        A_tensor, A2_tensor = self.get_X(paths[0], params, A.size, A2_tensor_)
        B_tensor, B2_tensor = self.get_X(paths[1], params, A.size, B2_tensor_)
        C_tensor = C2_tensor = D_tensor = D2_tensor = 0
        if (self.opt.isTrain):
            C2_tensor_, C = self.get_X2(paths[5], params)
            C_tensor, C2_tensor = self.get_X(paths[4], params, A.size, C2_tensor_)
            D2_tensor_, D = self.get_X2(paths[7], params)
            D_tensor, D2_tensor = self.get_X(paths[6], params, A.size, D2_tensor_)
        input_dict = {'A': A_tensor, 'A2': A2_tensor, 'B': B_tensor, 'B2': B2_tensor, 'C': C_tensor, 'C2': C2_tensor,
                'D': D_tensor, 'D2': D2_tensor, 'path': paths[0] + '_' + paths[1], 'same_style': self.same_style(paths[2], paths[3])}
        return input_dict

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)                
        A_scaled = transform_scaleA(self.crop(A_img))
        return A_scaled

    def get_face_image(self, A_path, transform_A, transform_L, size, img):
        # read face keypoints from path and crop face region
        keypoints, part_list, part_labels = self.read_keypoints(A_path, size)

        # draw edges and possibly add distance transform maps
        add_dist_map = not self.opt.no_dist_map
        im_edges, dist_tensor = self.draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map)
        
        # canny edge for background
        if not self.opt.no_canny_edge:
            edges = feature.canny(np.array(img.convert('L')))        
            edges = edges * (part_labels == 0)  # remove edges within face
            im_edges += (edges * 255).astype(np.uint8)
        edge_tensor = transform_A(Image.fromarray(self.crop(im_edges)))

        # final input tensor
        input_tensor = torch.cat([edge_tensor, dist_tensor]) if add_dist_map else edge_tensor
        label_tensor = transform_L(Image.fromarray(self.crop(part_labels.astype(np.uint8)))) * 255.0
        return input_tensor, label_tensor

    def read_keypoints(self, A_path, size):        
        # mapping from keypoints to face part 
        part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                     [range(60, 65), [64,65,66,67,60]]                 # tongue
                    ]
        label_list = [1, 2, 2, 3, 4, 4, 5, 6] # labeling for different facial parts        
        keypoints = np.loadtxt(A_path, delimiter=',')
        
        # add upper half face by symmetry
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0,1] + pts[-1,1]) / 2
        upper_pts = pts[1:-1,:].copy()
        upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1,:]))  

        # label map for facial part
        w, h = size
        part_labels = np.zeros((h, w), np.uint8)
        for p, edge_list in enumerate(part_list):                
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            cv2.fillPoly(part_labels, pts=[pts], color=label_list[p]) 

        return keypoints, part_list, part_labels

    def draw_face_edges(self, keypoints, part_list, transform_A, size, add_dist_map):
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        dist_tensor = 0
        e = 1                
        for edge_list in part_list:
            for edge in edge_list:
                im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                    drawEdge(im_edges, curve_x, curve_y)
                    if add_dist_map:
                        drawEdge(im_edge, curve_x, curve_y)
                                
                if add_dist_map: # add distance transform map on each facial part
                    im_dist = cv2.distanceTransform(255-im_edge, cv2.DIST_L1, 3)    
                    im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
                    im_dist = Image.fromarray(im_dist)
                    tensor_cropped = transform_A(self.crop(im_dist))                    
                    dist_tensor = tensor_cropped if e == 1 else torch.cat([dist_tensor, tensor_cropped])
                    e += 1

        return im_edges, dist_tensor

    def get_crop_coords(self, keypoints, size):                
        min_y, max_y = keypoints[:,1].min(), keypoints[:,1].max()
        min_x, max_x = keypoints[:,0].min(), keypoints[:,0].max()
        offset = (max_x - min_x) // 2
        min_y = max(0, min_y - offset*2)
        min_x = max(0, min_x - offset)
        max_x = min(size[0], max_x + offset)
        max_y = min(size[1], max_y + offset)
        self.min_y, self.max_y, self.min_x, self.max_x = int(min_y), int(max_y), int(min_x), int(max_x)        

    def crop(self, img):
        return img
        #???
        if isinstance(img, np.ndarray):
            return img[self.min_y:self.max_y, self.min_x:self.max_x]
        else:
            return img.crop((self.min_x, self.min_y, self.max_x, self.max_y))

    def __len__(self):
        return len(self.all_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'PoseDataset'
