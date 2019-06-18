import os
A_list = ['__Record024__Camera_5__170927_071032417_Camera_5']
A_list = ['_a']
B_list = ['__Record024__Camera_6__170927_070953498_Camera_6', '__Record031__Camera_5__170927_071846100_Camera_5', '__Record035__Camera_5__170927_072446711_Camera_5']
data_path = './datasets/apollo/'

def label(A):
    return os.path.join(data_path, 'label', 'Label' + A + '_bin.png')
def img(A):
    return os.path.join(data_path, 'img', 'ColorImage' + A + '.jpg')

lines = []
for A in A_list:
    for B in B_list:
        lines.append(label(A) + '&' + label(B) + '&' + img(B) + '&' + img(A) + '\n')
f = open(os.path.join(data_path, 'debug_list.txt'), 'w')
f.writelines(lines)
