import os
def expand(path):
    f = open(path, "r")
    fs = f.readlines()
    ans = []
    for line in fs:
        paths = line.rstrip('\n').split('&')
        new = [paths[0], paths[1], paths[2], paths[3], paths[1], paths[2], paths[1], paths[2]]
        ans.append('&'.join(new) + '\n')
    fw = open(path + '_', "w")
    fw.writelines(ans)
path = 'datasets/YouTubeFaces_/'
expand(os.path.join(path, 'val_list.txt'))
expand(os.path.join(path, 'test_list.txt'))
