import numpy as np
import os
import sys


root = sys.argv[1]

train = open(os.path.join(root, 'splits/train.csv'), 'r')
test = open(os.path.join(root, 'splits/val.csv'), 'r')
ground_path = os.path.join(root, 'streetview/panos')
semantic_path = os.path.join(root, 'streetview/panos_semantic')

train_lists = train.readlines()
test_lists = test.readlines()

trainset = []
for i, train_list in enumerate(train_lists):
    data = {}
    aerial_path = train_list.split(',')[0]
    data['aerial'] = os.path.join(root, aerial_path)
    im_num = os.path.basename(aerial_path).split('.')[0]
    grounds_path = os.path.join(ground_path, f'{im_num}.jpg')
    data['ground'] = grounds_path
    semantics_path = os.path.join(semantic_path, f'{im_num}.png')
    data['semantic'] = semantics_path
    trainset.append(data)

np.save(os.path.join(root,'splits/train.npy'), trainset)
np.savetxt(os.path.join(root,'splits/train.txt'), trainset, fmt='%s')


testset = []
for i, val_list in enumerate(test_lists):
    data = {}
    aerial_path = val_list.split(',')[0]
    data['aerial'] = os.path.join(root, aerial_path)
    im_num = os.path.basename(aerial_path).split('.')[0]
    grounds_path = os.path.join(ground_path, f'{im_num}.jpg')
    data['ground'] = grounds_path
    semantics_path = os.path.join(semantic_path, f'{im_num}.png')
    data['semantic'] = semantics_path
    testset.append(data)

np.save(os.path.join(root,'splits/test.npy'), testset)
np.savetxt(os.path.join(root,'splits/test.txt'), testset, fmt='%s')

