import re
import os
import cv2
import csv
import time
import random
import logging
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from Models.ContrastModel import ContrastNet, ContrastNet0, ContrastNet_center, ContrastNet_center_ca, ContrastNet1_center_ca

from GCN.gcn import Net
from GCN.dataset_gcn_animal import read_edge_index, Angle_relation
from torch_geometric.data import Data

from utils.dataset_csv import DatasetPoseCSV
from Eval.PCK_Mine import PCK_metric, PCK_metric_box
from Eval.CalAP import cal_AP, cal_OKS


def get_map(heatmap):
    tmp_map = np.max(heatmap, axis=0)  # H, W
    tmp_map = np.array(np.clip(tmp_map * 255, 0, 255), dtype=np.uint8)
    tmp_map = cv2.applyColorMap(tmp_map, cv2.COLORMAP_JET)  # H, W, C
    return tmp_map


def predict_img(net,
                source_img,
                target_img,
                device,
                resize_w,
                resize_h,
                scale_factor):
    net.eval()

    s_img = DatasetPoseCSV.preprocess(resize_w, resize_h, source_img, scale_factor, 1)
    s_img = torch.from_numpy(s_img)  # self.resize_w, self.resize_h, img0, self.scale, 1
    s_img = s_img.unsqueeze(0)
    s_img = s_img.to(device=device, dtype=torch.float32)

    t_img = DatasetPoseCSV.preprocess(resize_w, resize_h, target_img, scale_factor, 1)
    t_img = torch.from_numpy(t_img)  # self.resize_w, self.resize_h, img0, self.scale, 1
    t_img = t_img.unsqueeze(0)
    t_img = t_img.to(device=device, dtype=torch.float32)


    with torch.no_grad():
        output1, output2 = net(s_img, t_img)

        probs = F.softmax(output1, dim=1)
        probs = probs.squeeze(0)
        output1 = probs.cpu()

        output2 = torch.sigmoid(output2)
        output2 = output2.squeeze(0)
        output2 = output2.cpu()
    return output1, output2


def heatmap_to_points(Img, heatmap, numPoints, ori_W, ori_H):
    Img = cv2.resize(Img, (ori_W, ori_H))
    keyPoints = np.zeros([numPoints, 2])  # 第一行为y（对应行数），第二行为x（对应列数）
    # Img = cv2.resize(Img, (resize_w // 4, resize_h // 4))
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        # hm = heatmap[j, :, :]
        mx = np.max(hm)
        if mx > 0.059:  # 0.075paper
            center = np.unravel_index(np.argmax(hm), hm.shape)   # 寻找最大值点
            # print('center:', center)
            keyPoints[j, 0] = center[1]   # X
            keyPoints[j, 1] = center[0]   # Y
            # cv2.circle(Img, (center[1], center[0]), 1, (0, 0, 255), 2)  # 画出heatmap点重心
    return keyPoints


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


def draw_areamap(Img, Areamap, ori_W, ori_H, outfile, Filename):
    Img = cv2.resize(Img, (ori_W, ori_H)) * 0.5
    AP = cv2.resize(Areamap[0, :, :], (ori_W, ori_H))
    NormMap = (AP - np.mean(AP)) / (np.max(AP) - np.mean(AP))
    NormMap2 = (NormMap + np.abs(NormMap)) / 2
    AP = np.round(NormMap2 * 100)

    Img[:, :, 1] = AP + Img[:, :, 1]
    name = outfile + '/Areamap/' + Filename + '_Areamap.jpg'
    cv2.imwrite(name, Img)


def draw_relation2(Img, allPoints, relations):  # 640*480
    Img = cv2.resize(Img, (640, 480))
    for k in range(len(relations)):
        if max(allPoints[relations[k][0] - 1, :]) > 0 and max(allPoints[relations[k][1] - 1, :]) > 0:
            c_x1 = int(allPoints[relations[k][0]-1, 0] * (640/320))
            c_y1 = int(allPoints[relations[k][0]-1, 1] * (480/256))
            c_x2 = int(allPoints[relations[k][1]-1, 0] * (640/320))
            c_y2 = int(allPoints[relations[k][1]-1, 1] * (480/256))
            cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    r = np.arange(50, 255, int(205/len(allPoints)))

    for j in range(len(allPoints)):  # 在原图中画出关键点
        cv2.circle(Img, (int(allPoints[j, 0] * (640/320)), int(allPoints[j, 1] * (480/256))), 2,
                   [int(r[len(allPoints) - j]), 20, int(r[j])], 2)
    return Img


def read_labels(label_dir):
    # 读标签
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
        imgs_num = len(labels)
    print('imgs_num:', imgs_num)
    return labels, imgs_num


def img_augmentation(image):
    return cv2.flip(image, 1)  # 翻转 0表示上下，正数表示左右，负数表示上下左右都翻转


def inference(net,
              source_img,
              target_img,
              device,
              resize_w,
              resize_h,
              scale_factor,
              num_points):

    heatmap, areamap = predict_img(net=net,
                                   source_img=source_img,
                                   target_img=target_img,
                                   scale_factor=scale_factor,
                                   device=device,
                                   resize_w=resize_w,
                                   resize_h=resize_h)
    heatmap = heatmap.numpy()
    # areamap = areamap.numpy()
    keyPoints = heatmap_to_points(img0, heatmap, num_points, resize_w, resize_h)

    return keyPoints


def graph_inference(GNN, points1, points2, num_pointsnum_points):
    edge_index = read_edge_index(num_points)
    x = np.zeros([8, num_points * 2])
    x[0:2, 0:num_points] = points1.T
    x[0:2, num_points:] = points2.T

    # 计算夹角
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


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batch')

    parser.add_argument('--model', '-m',
                        default='./TrainedModels/ContrastiveModels/Center_TigDog/ContrastTigDog_ht/CP_epoch500.pth',
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--graph_model', '-g',
                        default='E:/Codes/Mine/WDA/GCN/TrainedGCN/model1_HorseTiger_noise0011_neg06/CP_epoch600.pth',
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str,
                        # default='./TrainedModels/ContrastiveModels/GAN/GAN_cam_Vertical_lr0005/generator_epoch2.pth',
                        # default='./TrainedModels/ContrastiveModels/center/ContrastCenter_SS/CP_epoch500.pth',
                        help='the path of backbone')
    parser.add_argument('-f', '--target-img', dest='target', type=str,
                        # default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/landmarks/horse/horse_test.csv',
                        default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/landmarks/tiger/tiger_test.csv',
                        help='Label path of target images')
    parser.add_argument('-t', '--source-path', dest='source', type=str,
                        default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/landmarks/horse/horse_test.csv',
                        # default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/landmarks/tiger/tiger_test.csv',
                        help='Label path of source image label')
    parser.add_argument('-i', '--img_dir', default='D:/Data/Pose/TigDog/behaviorDiscovery2.0/',
                        metavar='INPUT', nargs='+', help='filenames of input images')
    parser.add_argument('-o', '--output',
                        default='./results/ContrastiveModels/Center_TigDog/ContrastTigDog_ht/test_ST2/',
                        metavar='OUTPUT', nargs='+', help='Filenames of ouput images')
    parser.add_argument('-c', '--channel', default=19,
                        metavar='CHANNEL', nargs='+', help='Number of key points')
    parser.add_argument('--scale', '-s', type=float, help="Scale factor for the input images", default=1)
    return parser.parse_args()


if __name__ == "__main__":
    curve_name = 'TigDog_ContrastST_testST'
    args = get_args()
    in_files = args.img_dir
    out_files = args.output
    num_points = args.channel
    isExists = os.path.exists(out_files)
    if not isExists:  # 判断结果
        os.makedirs(out_files)

    resize_w = 320
    resize_h = 256
    extract_list = ["layer4"]
    if num_points == 14:
        keypoints = ['Right_ankle', 'Right_knee', 'Right_hip', 'Left_hip', 'Left_knee', 'Left_ankle',  'Right_wrist',
              'Right_elbow', 'Right_shoulder', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Neck', 'Head_top']  # LSP
    elif num_points == 17:
        keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                     'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    elif num_points == 6:
        keypoints = ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head']
        relation = [[1, 4], [1, 5], [2, 3], [2, 5], [3, 6], [4, 6]]

    elif num_points == 19:
        keypoints = ['leftEye', 'rightEye', 'chin', 'frontLeftHoof', 'frontRightHoof', 'backLeftHoof', 'backRightHoof'
                     'tailStart', 'frontLeftKnee', 'frontRightKnee', 'backLeftKnee', 'backRightKnee', 'leftShoulder',
                     'rightShoulder', 'frontLeftHip', 'frontRightHip', 'backLeftHip', 'backRightHip', 'neck']
        relation = [[0, 2], [0, 18], [1, 2], [1, 18], [3, 8], [4, 9], [5, 10], [6, 11], [7, 12], [7, 13], [7, 16],
                    [7, 17], [8, 14], [9, 15], [10, 16], [11, 17], [12, 14], [12, 18], [13, 15], [13, 18]]

    # 构建网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # net = ContrastNet0(None, extract_list, device, in_channel=3, nof_joints=num_points, train=False)
    # net = ContrastNet1_center_ca(args, None, extract_list, device, in_channel=3, nof_joints=num_points, train=False)
    net = ContrastNet_center(None, extract_list, device, in_channel=3, nof_joints=num_points, train=False)
    logging.info("Loading model {}".format(args.model))
    net.to(device=device)
    # 加载网络权重
    net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    logging.info("Model loaded !")

    # 图网络
    GraphNet = Net(8).to(device=device)
    GraphNet.load_state_dict(torch.load(args.graph_model, map_location=device), strict=False)
    logging.info("Graph model loaded !")

    # stage2阶段先加载RatNet在前面训练好的权重，然后再加载GAN训练的resnetnet部分权重
    # net.SubResnet.load_state_dict(torch.load(args.path_backbone, map_location=device), strict=False)
    # net.ResnetCam.SubResnet.load_state_dict(torch.load(args.path_backbone, map_location=device), strict=False)
    # net.ResnetCam.load_state_dict(torch.load(args.path_backbone, map_location=device), strict=False)
    # print('Pretrained generator weights have been loaded!')

    # 读取测试数据列表
    labels, imgs_num = read_labels(args.source)  # 读取标签
    target_labels, target_num = read_labels(args.target)

    points_all_gt = np.zeros([imgs_num - 1, num_points + 2, 2])  # 存放所有人的所有关键点标签
    points_all_pred = np.zeros([imgs_num - 1, num_points, 2])  # 存放检测到的所有关键点
    time_all = 0
    for k in range(1, imgs_num):
        # 读取图像
        img_path = labels[k][0]
        fn = os.path.join(in_files + img_path)

        img0 = cv2.imread(fn)
        imgf = cv2.flip(img0, 1)  # 翻转 0表示上下，正数表示左右，负数表示上下左右都翻转
        H, W, C = img0.shape

        # 读取target数据
        sort = random.sample(range(1, target_num), 1)[0]
        target_path = target_labels[sort][0]
        target_fn = os.path.join(in_files + target_path)
        target_img0 = cv2.imread(target_fn)

        # 网络预测
        keyPoints0 = inference(net=net,
                               source_img=img0,
                               target_img=target_img0,
                               scale_factor=args.scale,
                               device=device,
                               resize_w=resize_w,
                               resize_h=resize_h,
                               num_points=num_points)

        keyPoints_f = inference(net=net,
                               source_img=imgf,
                               target_img=target_img0,
                               scale_factor=args.scale,
                               device=device,
                               resize_w=resize_w,
                               resize_h=resize_h,
                               num_points=num_points)

        # 图推理
        pred = graph_inference(GraphNet, keyPoints0, keyPoints_f, num_points)
        # print(pred)

        # 保存图像
        img_0 = draw_relation2(img0, keyPoints0, relation)
        img_f = draw_relation2(imgf, keyPoints_f, relation)
        searchContext1 = '/'
        numList = [m.start() for m in re.finditer(searchContext1, labels[k][0])]
        filename = img_path[numList[0] + 1:numList[1]] + '_' + img_path[numList[1] + 1:]
        save_name = out_files + filename
        save_name_f = out_files + filename[:-4] + '_f_' + str(int(pred)) + '.jpg'
        if k % 100 == 0:
            print(k, img_path, save_name)
        cv2.imwrite(save_name, img_0)
        cv2.imwrite(save_name_f, img_f)

        # 读取标签，评估精度
        searchContext2 = '_'
        for n in range(num_points):
            numList = [m.start() for m in re.finditer(searchContext2, labels[k][n + 3])]
            point = [int(labels[k][n + 3][0:numList[0]]),
                     int(labels[k][n + 3][numList[0] + 1:numList[1]])]
            # resize
            point_resize = point
            point_resize[0] = point[0] * (resize_w / W)
            point_resize[1] = point[1] * (resize_h / H)
            points_all_gt[k - 1, n, :] = point_resize
        # 读取box信息
        numList = [m.start() for m in re.finditer(searchContext2, labels[k][2])]
        box = [int(labels[k][2][0:numList[0]]), int(labels[k][2][numList[0] + 1:numList[1]]),
               int(labels[k][2][numList[1] + 1:numList[2]]), int(labels[k][2][numList[2] + 1:])]
        box_resize = []
        box_resize.append(box[0] * (resize_w / W))
        box_resize.append(box[1] * (resize_h / H))
        box_resize.append(box[2] * (resize_w / W))
        box_resize.append(box[3] * (resize_h / H))
        box_resize = np.array(box_resize)
        points_all_gt[k - 1, num_points, :] = box_resize[0:2]
        points_all_gt[k - 1, num_points + 1, :] = box_resize[2:4]
        # print(labels[k][2], box, box_resize)

        points_all_pred[k - 1, :, :] = keyPoints0
        # print('pred:', points_all_pred, '   gt:', points_all_gt)

    time_mean = time_all / imgs_num
    print('mean time:', time_mean)
    # 保存标签和结果
    Save_result = out_files + 'results.mat'
    sio.savemat(Save_result, {'points_all_gt': points_all_gt, 'points_all_pred': points_all_pred})

    mask = np.ones([imgs_num - 1, num_points])
    thr = 0.5
    normalize = np.full([imgs_num - 1, 2], 20, dtype=np.float32)

    # pck_mine
    pred_name = out_files + 'points_all_pred.npy'
    gt_name = out_files + 'points_all_gt.npy'
    np.save(pred_name, points_all_pred)
    np.save(gt_name, points_all_gt)

    thr = np.linspace(0, 1, 101)
    mean_all = np.ones(101)
    for i in range(101):
        mean_points, var_points, mean_all[i], var_all = PCK_metric_box(points_all_pred, points_all_gt, 19, 20, thr[i])
        if thr[i] == 0.05:
            print('pck_points_mean:', mean_points)
            print('pck_points_val:', var_points)
            print('pck_all_mean:', mean_all[i], '    pck_all_val:', var_all)
    np.save(out_files + 'pck_mean101_' + curve_name + '.npy', mean_all)
    plt.plot(thr, mean_all, color='b')  # 绘制loss曲线
    plt.xlabel("normalized distance", fontsize=12)
    plt.ylabel("PCK", fontsize=12)
    plt.savefig(out_files + 'pcks_101' + curve_name + '.jpg')
    # print('PCK_mean_all:', mean_all)

    # 计算AP值
    OKS = cal_OKS(points_all_pred, points_all_gt, sigmas=0.055)
    AP, AP50, AP75 = cal_AP(OKS)
    print('  AP50:', AP50, ' AP75:', AP75, 'mAP:', AP, 'OKS:', np.mean(OKS))
