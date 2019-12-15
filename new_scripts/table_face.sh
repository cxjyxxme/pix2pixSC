### Adding instances and encoded features
python gen_vis_face.py --name label2city_256p_face_102 --label_nc 0 --test_delta_path 'vis_delta.txt' --no_canny_edge --input_nc 15 --fineSize 256 --dataroot './datasets/FaceForensics3/' --model 'c_pix2pixHD' --no_instance --loadSize 256 --phase 'val' --gpu_ids 0;
