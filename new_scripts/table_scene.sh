################################ Testing ################################
# first precompute and cluster all features
#python encode_features.py --name label2city_256p_feat;
# use instance-wise features

python gen_vis_bdd.py --name label2city_256p_face_108 --label_nc 42 --use_new_label --resize_or_crop 'scale_width_and_crop' --fineSize 256 --label_indexs '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,255' --dataroot './datasets/bdd2/' --model 'c_pix2pixHD' --gpu_ids 2 --no_instance --loadSize 256 --phase 'val';
