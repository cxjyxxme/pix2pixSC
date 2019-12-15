### Adding instances and encoded features
python infer_face.py --name label2city_256p_face_102 --label_nc 0 --no_canny_edge --input_nc 15 --fineSize 256 --dataroot './inference/data/' --model 'c_pix2pixHD' --no_instance --loadSize 256 --phase 'infer' --gpu_ids 0;
