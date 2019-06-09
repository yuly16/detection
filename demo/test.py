import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from spn import object_localization
import experiment.util as utils

DATA_ROOT = '/home/yuly/detection/SPN/data/voc/VOCdevkit/VOC2007'
ground_truth = utils.load_ground_truth_voc(DATA_ROOT, 'trainval')
print(ground_truth['gt_labels'][2])
model_path = './logs/voc2007/model.pth.tar'
model_dict = utils.load_model_voc(model_path, True)

import os
import torch
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from spn import object_localization
import experiment.util as utils

id2class = {}
for item in open('/home/yuly/multiclass/PascalVOC/categories.txt','r').readlines():
    item = item.strip()
    item = item.split(' ')
    id2class[int(item[0])] = item[1]

def init_dict(gtpath):
    print('building dict……')
    label_dict = [{} for _ in range(20)]
    fid = open(gtpath,'r')
    for item in fid.readlines():
        item = item.strip()
        item = item.split(' ')
        name = item[0]
        classid = int(item[1])
        x = float(item[2])
        y = float(item[3])
        w = float(item[4])
        h = float(item[5])
        if name not in label_dict[classid]:
            label_dict[classid][name] = []
        label_dict[classid][name].append([x,y,x+w,y+h,False])
    return label_dict

model_path = './logs/voc2007/model.pth.tar'
model_dict = utils.load_model_voc(model_path, True)
dataroot='/home/yuly/multiclass/PascalVOC/JPEGImages'
datpath = '/home/yuly/multiclass/PascalVOC/bonus_ground_truth.txt'
if not os.path.exists("groundtruths"):
    os.mkdir("groundtruths")
fid = open(datpath,'r')
for item in fid.readlines():
    item = item.strip()
    item = item.split(' ')
    name = item[0]
    classid = int(item[1])
    x = float(item[2])
    y = float(item[3])
    w = float(item[4])
    h = float(item[5])
    aline = [id2class[classid],str(x),str(y),str(x+w),str(y+h)]
    with open("groundtruths//" + name + ".txt", 'a') as fid:
        fid.write(' '.join(aline) + '\n')


label_dict = init_dict(datpath)
predictions = []
for i, clagt in tqdm(enumerate(label_dict)):
    for j in clagt:
        _, input_var = utils.load_image_voc('/home/yuly/multiclass/PascalVOC/JPEGImages/' + j + '.jpg')
        preds, labels = object_localization(model_dict, input_var, location_type='bbox', gt_labels=np.array([i]),obnum=len(clagt[j]),
                                            nms_threshold=0.7)
        # print('aaa')
        # print(item)
        # print(preds)
        predictions += [(i,) + p for p in preds]
        for pred in preds:
            aline = [id2class[i],str(pred[5]),str(pred[1]),str(pred[2]),str(pred[3]),str(pred[4])]
            if not os.path.exists("detections"):
                os.mkdir("detections")
            with open("detections/"+j+".txt",'a') as fid:
                fid.write(' '.join(aline)+'\n')



# with open(datpath) as fid:
#     predictions = []
#     gt = []
#
#     for item in tqdm(fid.readlines()):
#         item = item.strip()
#         item = item.split(' ')
#         name = item[0]
#         classid = int(item[1])
#         x = float(item[2])
#         y = float(item[3])
#         w = float(item[4])
#         h = float(item[5])
#         _, input_var = utils.load_image_voc('/home/yuly/multiclass/PascalVOC/JPEGImages/'+name+'.jpg')
#         preds, labels = object_localization(model_dict, input_var, location_type='bbox', gt_labels = np.array([classid]), nms_threshold=0.7)
#         print('aaa')
#         print(item)
#         print(preds)
#         predictions += [(name,) + p for p in preds]
#         gt += [(1,) + (classid,x,y,w+x,h+y)]