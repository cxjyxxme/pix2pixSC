from shutil import copyfile
tp = 'datasets/cityscapes/val_'
pre = 'aachen_000018_000019'
pre = 'bremen_000108_000019'
pre = 'frankfurt_000001_008688'
pre = 'frankfurt_000001_060906'
s_img_path = tp + 'img/' + pre + '_leftImg8bit.png'
t_img_path = 'datasets/test/img.png'
s_inst_path = tp + 'inst/' + pre + '_gtFine_instanceIds.png'
t_inst_path = 'datasets/test/inst.png'
s_label_path = tp + 'label/' + pre + '_gtFine_labelIds.png'
t_label_path = 'datasets/test/label.png'
copyfile(s_img_path, t_img_path)
copyfile(s_inst_path, t_inst_path)
copyfile(s_label_path, t_label_path)
