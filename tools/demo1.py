#!/usr/bin/env python
#coding=utf-8
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
#changed by zhangjunyi
#using opencv to visualize the object dection result
#2018/07/07
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import matplotlib.patches as patches

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets,ax,image_name="001150.JPG",a=1,thresh=0.5):
    """Draw detected bounding boxes."""
    print("heheheh")
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0
    if 1 == a:
        im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    else:
        im_file = os.path.join("./output",image_name)
    
    im = cv2.imread(im_file) 
    fig,ax = plt.subplots(1,1)

    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print(bbox[0], bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(im, (bbox[0], bbox[1]), ((bbox[2]),bbox[3]),(0, 0, 255), thickness=2)
        cv2.putText(im, class_name+"=:"+str(score), (int(bbox[2]-int((bbox[2]-bbox[0])/2)), int(bbox[3]-25)), font, 0.4, (255, 255, 255), 1)
        cv2.imwrite("./output/"+image_name,im)

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    a = 1
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        print(a)
        result=vis_detections(im, cls, dets, ax,image_name, a,thresh=CONF_THRESH) 
        if result !=0:    
    	   a = a + 1

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join('./data/output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))


    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(21,
                          tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))

    net.eval()
    if  torch.cuda.is_available():#只使用CPU.若想使用GPU在if后边加上not
        net._device = 'cpu'
    net.to(net._device)

    print('Loaded network {:s}'.format(saved_model))

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    im_names_1 = ['001150.jpg']#测试图片
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)

