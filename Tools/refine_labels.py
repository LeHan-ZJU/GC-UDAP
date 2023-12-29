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
from Models.ContrastModel import ResSelf_pre

from GCN.gcn import Net
# from GCN.dataset_gcn_animal import read_edge_index, Angle_relation
from GCN.dataset_gcn_regression import read_edge_index, Angle_relation_multiAngles, rotate_img
from torch_geometric.data import Data

from Tools.post_process import heatmap_to_points, draw_relation
from utils.dataset_csv import DatasetPoseCSV
from Eval.PCK_Mine import PCK_metric_box
from Eval.CalAP import cal_AP, cal_OKS


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

    Img = DatasetPoseCSV.preprocess(resize_w, resize_h, Img, 1)
    Img = torch.from_numpy(Img)  # self.resize_w, self.resize_h, img0, self.scale, 1
    Img = Img.unsqueeze(0)
    Img = Img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output1 = net(Img)

        probs = F.softmax(output1, dim=1)
        probs = probs.squeeze(0)
        output1 = probs.cpu()

    return output1


def read_labels(label_dir):
    # 读标签
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
        imgs_num = len(labels)
    print('imgs_num:', imgs_num)
    return labels, imgs_num, labels[0]


def img_augmentation(image):
    return cv2.flip(image, 1)  # 翻转 0表示上下，正数表示左右，负数表示上下左右都翻转


def keypoints_inference(net,
                        Img,
                        device,
                        resize_w,
                        resize_h,
                        scale_factor,
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

    # 计算夹角
    # angles1 = Angle_relation(x[0:2, 0:num_points], edge_index.T, num_points)
    # angles2 = Angle_relation(x[0:2, num_points:], edge_index.T, num_points)
    # x[2, 0:num_points] = angles1
    # x[2, num_points:] = angles2
    angles1 = Angle_relation_multiAngles(x[:, 0:num_points], edge_index.T, num_points)
    angles2 = Angle_relation_multiAngles(x[:, num_points:], edge_index.T, num_points)
    x[2:, 0:num_points] = angles1
    x[2:, num_points:] = angles2

    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x.t(), edge_index=edge_index.t())

    GNN.eval()
    data = data.to(device)
    print(data.x, data.edge_index)
    output = GNN(data.x, data.edge_index, data.batch)

    pred = (output > 0.6).float()
    print(output, pred)

    return np.array(pred.item())


def inference(fn,
              net,
              device,
              resize_w,
              resize_h,
              num_points,
              GraphNet):

    img0 = cv2.imread(fn)
    H, W, C = img0.shape
    # imgf = cv2.flip(img0, args.flip)  # 翻转 0表示上下，正数表示左右，负数表示上下左右都翻转
    # imgf = cv2.rotate(img0, cv2.ROTATE_180)
    imgf = rotate_img(img0, 180, [W / 2, H / 2])

    # 网络预测
    keyPoints0 = keypoints_inference(net=net,
                                     img=img0,
                                     device=device,
                                     resize_w=resize_w,
                                     resize_h=resize_h,
                                     num_points=num_points)

    keyPoints_f = keypoints_inference(net=net,
                                      img=imgf,
                                      device=device,
                                      resize_w=resize_w,
                                      resize_h=resize_h,
                                      num_points=num_points)

    # 图推理
    graph_score = graph_inference(GraphNet, keyPoints0, keyPoints_f, num_points)

    return keyPoints0, keyPoints_f, graph_score, img0, imgf


def write_results(img0,
                  imgf,
                  graph_score,
                  keyPoints0,
                  keyPoints_f,
                  keypointsWriter,
                  keypointsWriter2,
                  resize_w,
                  resize_h,
                  label):
    H, W, C = img0.shape

    numList = [m.start() for m in re.finditer('/', label[0])]
    filename = img_path[numList[0] + 1:numList[1]] + '_' + img_path[numList[1] + 1:]

    if graph_score == 1:
        # print('keyPoints:', keyPoints0.shape, keyPoints0, keyPoints_f)
        # keyPoints0, keyPoints_f = refine_points(keyPoints0, keyPoints_f, resize_w, resize_h, args.flip)
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
        keypointsWriter2.writerow(keypointsRow)
        keypointsWriter2.writerow(keypointsRow_f)

        save_name = out_files + 'Right/' + filename
        save_name_f = out_files + 'Right/' + filename[:-4] + '_f' + str(args.flip) + '.jpg'

        # 读取标签，评估精度
        searchContext2 = '_'
        points_label = np.zeros([num_points + 2, 2])
        for n in range(num_points):
            numList = [m.start() for m in re.finditer(searchContext2, labels[k][n + 3])]
            point = [int(labels[k][n + 3][0:numList[0]]),
                     int(labels[k][n + 3][numList[0] + 1:numList[1]])]
            # resize
            point_resize = point
            point_resize[0] = point[0] * (resize_w / W)
            point_resize[1] = point[1] * (resize_h / H)
            # points_all_gt[k - 1, n, :] = point_resize
            points_label[n, :] = point_resize
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
        # points_all_gt[k - 1, num_points, :] = box_resize[0:2]
        # points_all_gt[k - 1, num_points + 1, :] = box_resize[2:4]
        points_label[num_points, :] = box_resize[0:2]
        points_label[num_points + 1, :] = box_resize[2:4]
        points_all_gt1.append(points_label)

        # points_all_pred[k - 1, :, :] = keyPoints0
        points_all_pred1.append(keyPoints0)
        # print('pred:', points_all_pred, '   gt:', points_all_gt)
    else:
        keypointsRow = [img_path, str(1), '0_0_0_0']
        for p in range(num_points):
            keypointsRow.append('0_0_0')
        keypointsWriter.writerow(keypointsRow)
        save_name = out_files + 'False/' + filename
        save_name_f = out_files + 'False/' + filename[:-4] + '_f0.jpg'

    # 保存图像
    if k % 100 == 0:
        print(k, img_path, save_name)

    img_0 = draw_relation(img0, keyPoints0, relation)
    img_f = draw_relation(imgf, keyPoints_f, relation)
    cv2.imwrite(save_name, img_0)
    cv2.imwrite(save_name_f, img_f)


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
    parser.add_argument('-n2', '--num_points2', metavar='N', type=int, default=6,
                        help='Number of keypoints', dest='num_points2')

    parser.add_argument('--graph_model', '-g', metavar='FILE',
                        default='./GCN/TrainedGCN/3classes/Reg5_3edg_RatPoseAll_noise005_neg06/CP_epoch300.pth',
                        # default='./GCN/TrainedGCN/RegModel1_Vertical_noise0005_neg07/CP_epoch300.pth',
                        # default='./GCN/TrainedGCN/model1_Vertical_noise0005_neg07/CP_epoch300.pth',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--model', '-m', metavar='FILE', help="Specify the file in which the model is stored",
                        default='./TrainedModels/ContrastiveModels/pretrain_ResSelf/debug1/CP_epoch300.pth',  # 6
                        # default='./TrainedModels/ContrastiveModels/pretrain_ResSelf/debug_VerticalTilt/CP_epoch300.pth'
                        )
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str, default=False,
                        # default='./TrainedModels/ContrastiveModels/GAN/GAN_cam_Vertical_lr0005/generator_epoch2.pth',
                        # default='./TrainedModels/ContrastiveModels/center/ContrastCenter_SS/CP_epoch500.pth',
                        help='the path of backbone')
    parser.add_argument('-t', '--label-path', dest='label', type=str,
                        # default='G:/Data/RatPose/label/RatPoseLabels/RatPoseAll_indoor_test.csv',
                        default='G:/Data/RatPose/label/RatPoseLabels/RatPoseAll_indoor_crop_v2_test.csv',
                        # default='G:/Data/RatPose/label/RatPoseLabels/Vertical_Indoor_label_new_test.csv',
                        # default='G:/Data/RatPose/label/RatPoseLabels/Vertical_Indoor_label_new_crop_v2_test.csv',
                        # default='G:/Data/RatPose/label/RatPoseLabels/VerticalTilt_crop_v2.csv',
                        #default='G:/Data/RatPose/label/RatPoseLabels/DataALL_v23_crop_v2_test.csv',
                        help='Label path of source image label')
    parser.add_argument('-i', '--img_dir', default='G:/Data/RatPose/all/', metavar='INPUT', nargs='+',
                        help='filenames of input images')
    parser.add_argument('-o', '--output', metavar='OUTPUT', nargs='+', help='Filenames of ouput images',
                        default='./results/ContrastiveModels/pretrain_ResSelf/debug1/testVerticalTilt_crop_v2all_RegM2/debuggraph/'
                        )
    parser.add_argument('--scale', '-s', type=float, help="Scale factor for the input images", default=4)
    parser.add_argument('--flip', '-f', type=int, default=1,
                        help="1_flip horizontal, 0_flip vertical, -1_horizontal&vertical")
    return parser.parse_args()


if __name__ == "__main__":
    curve_name = 'Rat_pretrain_ResSelf'
    args = get_args()
    in_files = args.img_dir
    out_files = args.output
    num_points = args.num_points
    isExists = os.path.exists(out_files + 'Right/')

    # 读取测试数据列表
    labels, imgs_num, label_head = read_labels(args.label)  # 读取标签

    if not isExists:  # 创建结果文件夹
        # os.makedirs(out_files)
        os.makedirs(out_files + 'Right/')
        os.makedirs(out_files + 'False/')

    # 生成csv，写表头
    csvFile = open(out_files + 'Right/Pesudo-label_all.csv', 'w+', newline='')
    keypointsWriter = csv.writer(csvFile)
    csvFile2 = open(out_files + 'Right/Pesudo-label.csv', 'w+', newline='')
    keypointsWriter2 = csv.writer(csvFile2)
    firstRow = label_head[:3+num_points]
    keypointsWriter.writerow(firstRow)
    keypointsWriter2.writerow(firstRow)


    resize_w = 640
    resize_h = 480
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

    net = ResSelf_pre(args, extract_list, device, train=False, nof_joints=num_points)
    logging.info("Loading model {}".format(args.model))
    net.to(device=device)
    # 加载网络权重
    net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    logging.info("Model loaded !")

    # 图网络
    GraphNet = Net(8, reg=True).to(device=device)
    GraphNet.load_state_dict(torch.load(args.graph_model, map_location=device), strict=False)
    logging.info("Graph model loaded !")

    # points_all_gt = np.zeros([imgs_num - 1, num_points + 2, 2])  # 存放所有人的所有关键点标签
    # points_all_pred = np.zeros([imgs_num - 1, num_points, 2])  # 存放检测到的所有关键点
    points_all_gt1 = []
    points_all_pred1 = []
    time_all = 0
    for k in range(1, imgs_num):
        # 读取图像
        img_path = labels[k][0]
        fn = os.path.join(in_files + img_path)

        keyPoints0, keyPoints_f, pred, Img0, Imgf = inference(fn, net,
              device,
              resize_w,
              resize_h,
              scale_factor,
              num_points,
              GraphNet)


        points_all_pred1, points_all_gt1 = write_results(Img0,
                                                         Imgf,
                                                         pred,
                                                         keyPoints0,
                                                         keyPoints_f,
                                                         keypointsWriter,
                                                         keypointsWriter2,
                                                         resize_w,
                                                         resize_h,
                                                         labels[k])

    time_mean = time_all / imgs_num
    print('mean time:', time_mean)
    # 保存标签和结果
    points_all_gt1 = np.array(points_all_gt1)
    points_all_pred1 = np.array(points_all_pred1)
    Save_result = out_files + 'results1.mat'
    sio.savemat(Save_result, {'points_all_gt1': points_all_gt1, 'points_all_pred1': points_all_pred1})

    mask = np.ones([imgs_num - 1, num_points])
    thr = 0.5
    normalize = np.full([imgs_num - 1, 2], 20, dtype=np.float32)

    # pck_mine
    pred_name = out_files + 'points_all_pred1.npy'
    gt_name = out_files + 'points_all_gt1.npy'
    np.save(pred_name, points_all_pred1)
    np.save(gt_name, points_all_gt1)

    thr = np.linspace(0, 1, 101)
    mean_all = np.ones(101)
    for i in range(101):
        mean_points, var_points, mean_all[i], var_all = PCK_metric_box(points_all_pred1, points_all_gt1, 6, 7, thr[i])
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
    OKS = cal_OKS(points_all_pred1, points_all_gt1, sigmas=0.055)
    AP, AP50, AP75 = cal_AP(OKS)
    print('  AP50:', AP50, ' AP75:', AP75, 'mAP:', AP, 'OKS:', np.mean(OKS))
