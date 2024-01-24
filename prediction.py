import re
import os
import cv2
import csv
import logging
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from Models.ContrastModel import ResSelf_pre

from utils.dataset_csv import DatasetPoseCSV


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cls_number', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--pretrained', default=False, action='store_false')
    parser.add_argument('--anchor_number', '-a', type=int, default=49)
    parser.add_argument('-b', '--batch', metavar='B', type=int, nargs='?', default=16, help='Batch size', dest='batch')
    parser.add_argument('-n', '--num_points', metavar='N', type=int, default=10,
                        help='Number of keypoints (net)', dest='num_points')

    parser.add_argument('--model', '-m', default='', metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str, default=False, help='the path of backbone')
    parser.add_argument('-t', '--label-path', dest='label', type=str, default='./path of test label/',
                        help='Label path of source domain')
    parser.add_argument('-i', '--img_dir', default='./path of imgs/',
                        metavar='INPUT', nargs='+', help='filenames of input images')
    parser.add_argument('-o', '--output', default='./results/', metavar='OUTPUT', nargs='+', help='Filenames of ouput images')
    return parser.parse_args()


def predict_img(net,
                Img,
                device,
                resize_w,
                resize_h):
    net.eval()

    Img = DatasetPoseCSV.preprocess(resize_w, resize_h, Img, 1)
    Img = torch.from_numpy(Img)
    Img = Img.unsqueeze(0)
    Img = Img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output1 = net(Img)

        probs = F.softmax(output1, dim=1)
        probs = probs.squeeze(0)
        output1 = probs.cpu()

    return output1


def heatmap_to_points(Img, heatmap, numPoints, ori_W, ori_H):
    # Transform heatmap to keyPoint coordinates
    Img = cv2.resize(Img, (ori_W, ori_H))
    keyPoints = np.zeros([numPoints, 2]) 
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        center = np.unravel_index(np.argmax(hm), hm.shape)   # find max value in heatmap
        keyPoints[j, 0] = center[1]   # X
        keyPoints[j, 1] = center[0]   # Y
    return Img, keyPoints


def draw_relation(Img, allPoints, relations):
    # Plot the result on the image
    Img = cv2.resize(Img, (640, 480))
    for k in range(len(relations)):
        c_x1 = int(allPoints[relations[k][0]-1, 0] * (640/resize_w))
        c_y1 = int(allPoints[relations[k][0]-1, 1] * (480/resize_h))
        c_x2 = int(allPoints[relations[k][1]-1, 0] * (640/resize_w))
        c_y2 = int(allPoints[relations[k][1]-1, 1] * (480/resize_h))
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    r = np.arange(50, 255, int(205/len(allPoints)))

    for j in range(len(allPoints)):
        cv2.circle(Img, (int(allPoints[j, 0] * (640/resize_w)), int(allPoints[j, 1] * (480/resize_h))), 2,
                   [int(r[len(allPoints) - j]), 20, int(r[j])], 2)
    return Img


def read_labels(label_dir):
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
        imgs_num = len(labels)
    print('imgs_num:', imgs_num)
    return labels, imgs_num


if __name__ == "__main__":
    args = get_args()
    in_files = args.img_dir
    out_files = args.output
    num_points = args.num_points
    isExists = os.path.exists(out_files)
    if not isExists:
        os.makedirs(out_files)

    resize_w = 640
    resize_h = 480
    extract_list = ["layer4"]
    
    # UDAP9.4K (Platform scenarios)
    if args.num_points == 6:
        keypoints = ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head']
        relation = [[1, 4], [1, 5], [2, 3], [2, 5], [3, 6], [4, 6]]
    
    # UDAP9.4K (Lawn scenarios)
    elif args.num_points == 10:
        keypoints = ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head', 'neck', 'spine', 'tail_middle', 'tail_end']
        relation = [[1, 5], [1, 8], [1, 7], [2, 5], [2, 7], [2, 8], [3, 7], [3, 6],
                    [4, 6], [4, 7], [5, 8], [7, 8], [5, 9], [9, 10]]
    
    # TigDog
    elif args.num_points == 19:
        keypoints = ['leftEye', 'rightEye', 'chin', 'frontLeftHoof', 'frontRightHoof', 'backLeftHoof', 'backRightHoof',
                     'tailStart', 'frontLeftKnee', 'frontRightKnee', 'backLeftKnee', 'backRightKnee', 'leftShoulder',
                     'rightShoulder', 'frontLeftHip', 'frontRightHip', 'backLeftHip', 'backRightHip', 'neck']
        relation = [[1, 2], [1, 3], [2, 3], [8, 13], [8, 14],  [13, 14], [13, 19], [14, 19], 
                    [13, 19], [15, 9], [15, 4], [14, 10], [16, 10], [16, 5], [8, 11], [17, 11], [17, 6],
                    [8, 12], [18, 12], [18, 7]]
    
    # AP10K
    elif args.num_points == 17:
        keyPoints = ['leftEye', 'rightEye', 'nose', 'neck', 'TailRoot', 'leftShoulder', 'leftElbow', 'leftFrontPaw', 
                     'rightShoulder', 'rightElbow', 'rightFrontPaw', 'leftHip', 'leftKnee', 'leftBackPaw', 'rightHip',
                     'rightKnee', 'rightBackPaw']
        relation = [[1, 2], [1, 3], [1, 6], [2, 3], [2, 9], [3, 4], [4, 5], [4, 6], [4, 9], [5, 12], 
                    [5, 15], [6, 7], [6, 9], [7, 8], [7, 10], [8, 11], [8, 14], [9, 10], [10, 11], 
                    [11, 17], [12, 13], [12, 15], [13, 14], [13, 16], [14, 17], [15, 16], [16, 17]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = ResSelf_pre(args, extract_list, device, train=False, nof_joints=num_points)

    logging.info("Loading model {}".format(args.model))
    net.to(device=device)

    # load weights
    if args.model:   
        # pretrained model
        net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
        print("Model loaded !")
    else:           
        # DAT models
        net.SubResnet.load_state_dict(torch.load(
            './TrainedModels/DAT/FeatureExtractor.pth',
            map_location=device), strict=False)
        print("SubResnet Model loaded !")
        net.FeatureAlignment.load_state_dict(torch.load(
            './TrainedModels/DAT/FeatureAlignment.pth',
            map_location=device), strict=False)
        print("FeatureAlignment Model loaded !")
        net.PoseHead.load_state_dict(torch.load(
            './TrainedModels/DAT/Decoder.pth',
            map_location=device), strict=False)
        print("PoseHead Model loaded !")

    labels, imgs_num = read_labels(args.label) 
    for k in range(1, imgs_num):
        img_path = labels[k][0]
        fn = os.path.join(in_files + img_path)

        img0 = cv2.imread(fn)
        H, W, C = img0.shape

        # inference
        heatmap1 = predict_img(net=net,
                               Img=img0,
                               device=device,
                               resize_w=resize_w,
                               resize_h=resize_h)

        heatmap1 = heatmap1.numpy()

        img, keyPoints = heatmap_to_points(img0, heatmap1, num_points, resize_w, resize_h)
        img = draw_relation(img0, keyPoints, relation)

        # save
        searchContext1 = '/'
        numList = [m.start() for m in re.finditer(searchContext1, labels[k][0])]
        filename = img_path[numList[0] + 1:numList[1]] + '_' + img_path[numList[1] + 1:]
        save_name = out_files + filename
        if k % 100 == 0:
            print(k, img_path, save_name)
        cv2.imwrite(save_name, img)

