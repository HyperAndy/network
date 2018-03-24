import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import shutil
from PIL import Image

homedir = os.getcwd()
base = './test/base/'
annotations = 'Annotations/'
images = 'Image/'
train = pd.read_csv(base+annotations+'label.csv',header=None)
train.columns = ['image_id', 'classes', 'label']

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
             'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
             'pant_length_labels']

#将各个图片分别归类到相应文件夹中（其中m均转化为n，后期再进行改进）
for i in range(len(classes)):
    cur_class = classes[i]  #当前处理类别
    
    print('......处理数据集:',cur_class)
    df_load = train[(train['classes'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']
    labels_new = []
    for labels in df_load['label']:
        labels = labels.replace('m','n') #将m用n替换
        labels_new.append(labels)
    df_load['label'] = labels_new
    n = len(df_load)
    n_class = len(df_load['label'][0])#类别数

    for label, group in df_load.groupby(['label']):
        print('......生成数据文件夹：', label)
        print('......图片个数：', group.shape[0])
        group_id = group['image_id']
#    group_label = label
        for image_path in group_id:
#        shutil.copy(base+image_path, 'home/u12292/'+image_path[:28]+label+'/'+image_path[28:])
            img = Image.open(base+image_path)
            folder = image_path.split('/')
            path_save = homedir+'/test/base/'+folder[0]+'/'+folder[1]+'/'+label
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            img.save(os.path.join(path_save, os.path.basename(image_path)))
        print('......数据文件夹生成完毕')
    print('......数据集处理完毕')
