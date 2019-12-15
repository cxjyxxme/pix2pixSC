### Adding instances and encoded features
python test_pose.py --name label2city_256p_pose_106 --how_many 2500 --label_nc 0 --test_delta_path "vis_delta.txt" --no_canny_edge --input_nc 23 --do_pose_dist_map --fineSize 256 --dataroot './datasets/YouTubePose2/' --model 'c_pix2pixHD' --no_instance --loadSize 256 --phase 'test_final' --gpu_ids 3;
