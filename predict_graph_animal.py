import re
import os
import cv2
import csv
import random
import logging
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from Models.ContrastModel import ResSelf_pre

from GCN.gcn import Net
from GCN.dataset_gcn_animal import read_edge_index, Angle_relation
from GCN.dataset_gcn_regression import rotate_img
from torch_geometric.data import Data

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
    parser.add_argument('-b', '--batch', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batch')
    parser.add_argument('-n', '--num_points', metavar='N', type=int, default=6,
                        help='Number of keypoints', dest='num_points')

    parser.add_argument('--graph_model', '-g', metavar='FILE', default='./TrainedGCL/model.pth',
                        help="path of the trained SGCL model")
    parser.add_argument('--model', '-m', metavar='FILE', default='./TrainedModels/model.pth',
                        help="path of the pretrained pose model")
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str, default=False,
                        help='the path of backbone')
    parser.add_argument('-t', '--label-path', dest='label', type=str,
                        default='./path of target image name list/',
                        help='Label path of source image label')
    parser.add_argument('-i', '--img_dir', default='./path of imgs/', metavar='INPUT', nargs='+',
                        help='filenames of input images')
    parser.add_argument('-o', '--output', metavar='OUTPUT', nargs='+', default='./results/', 
                        help='Filenames of ouput images')
    parser.add_argument('--scale', '-s', type=float, help="Scale factor for the input images", default=4)
    parser.add_argument('--flip', '-f', type=int, default=1,
                        help="1_flip horizontal, 0_flip vertical, -1_horizontal&vertical")
    return parser.parse_args()


def get_map(heatmap):
    tmp_map = np.max(heatmap, axis=0)  # H, W
    tmp_map = np.array(np.clip(tmp_map * 255, 0, 255), dtype=np.uint8)
    tmp_map = cv2.applyColorMap(tmp_map, cv2.COLORMAP_JET)  # H, W, C
    return tmp_map


def predict_img(net,
                Img,
                device,
                resize_w,
                resize_h):
    net.eval()

    Img = DatasetPoseCSV.preprocess(resize_w, resize_h, Img, trans=1)
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
    # Img = cv2.resize(Img, (ori_W, ori_H))
    keyPoints = np.zeros([numPoints, 2])  # 第一行为y（对应行数），第二行为x（对应列数）
    # Img = cv2.resize(Img, (resize_w // 4, resize_h // 4))
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        if max(max(row) for row in hm) < 0.1:
            print('max:', max(max(row) for row in hm))
        center = np.unravel_index(np.argmax(hm), hm.shape)   # 寻找最大值点
        # print('center:', center)
        keyPoints[j, 0] = center[1] * ori_W / resize_w   # X
        keyPoints[j, 1] = center[0] * ori_H / resize_h  # Y
        # cv2.circle(Img, (center[1], center[0]), 1, (0, 0, 255), 2)  # 画出heatmap点重心 img = img*0.3 print(keyPoints)
    return keyPoints


def cal_maxDis(p0):
    max_d = 0
    for i in range(p0.shape[0]):
        d = np.norm(p0[i, :])
        if d > max_d:
            max_d = d
    return max_d


def draw_heatmap(Img, heatmap, numPoints, ori_W, ori_H, outfile, Filename):
    Img = cv2.resize(Img, (ori_W, ori_H)) * 0.5
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        NormMap = (hm - np.mean(hm)) / (np.max(hm) - np.mean(hm))
        NormMap2 = (NormMap + np.abs(NormMap)) / 2
        map = np.round(NormMap2 * 100)
        # map = cv2.resize(map, (ori_W, ori_H))
        Img[:, :, 1] = map + Img[:, :, 1]

        name = outfile + Filename + '_' + keypoints[j] + '.jpg'
        cv2.imwrite(name, Img)
    return 0


def draw_relation2(Img, allPoints, relations):  # 640*480
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
    return labels, imgs_num, labels[0]


def img_augmentation(image):
    return cv2.flip(image, 1)


def inference(net,
              Img,
              device,
              resize_w,
              resize_h,
              num_points):

    heatmap = predict_img(net=net,
                          Img=Img,
                          device=device,
                          resize_w=resize_w,
                          resize_h=resize_h)

    heatmap = heatmap.numpy()
    # areamap = areamap.numpy()
    keyPoints = heatmap_to_points(Img, heatmap, num_points, resize_w, resize_h)

    return keyPoints


def graph_inference(GNN, points1, points2, num_points):
    edge_index = read_edge_index(num_points)
    x = np.zeros([8, num_points * 2])
    x[0:2, 0:num_points] = points1.T
    x[0:2, num_points:] = points2.T

    angles1 = Angle_relation(x[0:2, 0:num_points], edge_index.T, num_points)
    angles2 = Angle_relation(x[0:2, num_points:], edge_index.T, num_points)
    x[2, 0:num_points] = angles1
    x[2, num_points:] = angles2

    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x.t(), edge_index=edge_index.t())

    GNN.eval()
    data = data.to(device)
    output = GNN(data.x, data.edge_index, data.batch)

    pred = (output > 0.5).float()

    return np.array(pred.item())


if __name__ == "__main__":
    args = get_args()
    in_files = args.img_dir
    out_files = args.output
    num_points = args.num_points
    isExists = os.path.exists(out_files + 'Right/')

    labels, imgs_num, label_head = read_labels(args.label)

    if not isExists:
        os.makedirs(out_files + 'Right/')  # right prediction
        os.makedirs(out_files + 'False/')  # false prediction

    # csv init
    csvFile = open(out_files + 'Right/Pesudo-label.csv', 'w+', newline='')
    keypointsWriter = csv.writer(csvFile)
    csvFile2 = open(out_files + 'Right/noPesudo-label.csv', 'w+', newline='')
    keypointsWriter2 = csv.writer(csvFile2)
    firstRow = label_head[:3+num_points]
    keypointsWriter.writerow(firstRow)
    keypointsWriter2.writerow(firstRow)


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

    GraphNet = Net(8, reg=True).to(device=device)
    GraphNet.load_state_dict(torch.load(args.graph_model, map_location=device), strict=False)
    print("Graph model loaded !")

    for k in range(1, imgs_num):
        img_path = labels[k][0]
        fn = os.path.join(in_files + img_path)

        img0 = cv2.imread(fn)
        H, W, C = img0.shape
        imgf = rotate_img(img0, 180, [W/2, H/2])

        # Pose inference
        keyPoints0 = inference(net=net,
                               Img=img0,
                               device=device,
                               resize_w=resize_w,
                               resize_h=resize_h,
                               num_points=num_points)

        keyPoints_f = inference(net=net,
                                Img=imgf,
                                device=device,
                                resize_w=resize_w,
                                resize_h=resize_h,
                                num_points=num_points)

        # Graph inference
        pred = graph_inference(GraphNet, keyPoints0, keyPoints_f, num_points)

        searchContext1 = '/'
        numList = [m.start() for m in re.finditer(searchContext1, labels[k][0])]
        filename = img_path[numList[0] + 1:numList[1]] + '_' + img_path[numList[1] + 1:]

        if pred == 1:

            keypointsRow = [img_path, str(1),
                            str(int(min(keyPoints0[:, 0] * (W / resize_w)))) + '_' +
                            str(int(min(keyPoints0[:, 1] * (H / resize_h)))) + '_' +
                            str(int(max(keyPoints0[:, 0] * (W / resize_w)))) + '_' +
                            str(int(max(keyPoints0[:, 1] * (H / resize_h))))]
            keypointsRow_f = [img_path[:-4] + '_flip' + str(args.flip) + '.jpg', str(1),
                              str(int(min(keyPoints_f[:, 0] * (W / resize_w)))) + '_' +
                              str(int(min(keyPoints_f[:, 1] * (H / resize_h)))) + '_' +
                              str(int(max(keyPoints_f[:, 0] * (W / resize_w)))) + '_' +
                              str(int(max(keyPoints_f[:, 1] * (H / resize_h))))]

            for p in range(num_points):
                keypointsRow.append(str(int(keyPoints0[p, 0] * (W / resize_w))) + '_' +
                                    str(int(keyPoints0[p, 1] * (H / resize_h))) + '_' + str(0))
                keypointsRow_f.append(str(int(keyPoints_f[p, 0] * (W / resize_w))) + '_' +
                                      str(int(keyPoints_f[p, 1] * (H / resize_h))) + '_' + str(0))
            keypointsWriter.writerow(keypointsRow)
            keypointsWriter.writerow(keypointsRow_f)

            save_name = out_files + 'Right/' + filename
            save_name_f = out_files + 'Right/' + filename[:-4] + '_f' + str(args.flip) + '.jpg'

        else:
            keypointsRow = [img_path, str(1), '0_0_0_0']
            for p in range(num_points):
                keypointsRow.append('0_0_0')
            keypointsWriter2.writerow(keypointsRow)
            save_name = out_files + 'False/' + filename
            save_name_f = out_files + 'False/' + filename[:-4] + '_f0.jpg'

        # save image
        if k % 100 == 0:
            print(k, img_path, save_name)

        img_0 = draw_relation2(img0, keyPoints0, relation)
        img_f = draw_relation2(imgf, keyPoints_f, relation)
        cv2.imwrite(save_name, img_0)
        cv2.imwrite(save_name_f, img_f)

