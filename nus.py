import torch.utils.data as data
import pickle
import xlrd
from get_img_list import *
import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
from util import *




def category_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx

class NUSWIDE(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        
        #category = table.col_values(0)
        # category = category[925:]
        #self.cat2ind = category_to_idx(category)
        self.inp_name = inp_name

        result = pickle.load(open('/home/scnu_1/erzu/xufeiteng/ML_GCN/data/nus/nuswide_glove_word2vec.pkl', 'rb'))
        self.inp = result['class_embed']
        # index_81 = [i for i in range(925, 1006)]
        # self.inp = self.inp[[index_81]]
        self.num_classes = 81  #81 or 1006
        self.transform = transform
        '''
        if self.phase=='train':
            self.img_list = get_train_imglist()
        if self.phase == 'val':
            self.img_list = get_test_imglist()
        '''
        if self.phase == 'train81':
            self.img_list = get_train81_imglist()
        if self.phase == 'val81':
            self.img_list = get_test81_imglist()



    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item) 

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target

