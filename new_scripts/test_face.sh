### Adding instances and encoded features
python test_face.py --name label2city_256p_face_102 --how_many 2500 --label_nc 0 --no_canny_edge --input_nc 15 --fineSize 256 --dataroot './datasets/FaceForensics3/' --model 'c_pix2pixHD' --no_instance --loadSize 256 --phase 'test_final' --gpu_ids 0;
