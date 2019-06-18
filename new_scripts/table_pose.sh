### Adding instances and encoded features
python gen_vis_pose.py --name label2city_256p_pose_106 --label_nc 0 --test_delta_path "vis_delta.txt" --no_canny_edge --input_nc 23 --do_pose_dist_map --fineSize 256 --dataroot './datasets/YouTubePose2/' --model 'c_pix2pixHD' --no_instance --loadSize 256 --phase 'val' --gpu_ids 1;
