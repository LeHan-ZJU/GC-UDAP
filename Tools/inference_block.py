import os
import re
import csv
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from utils.dataset_csv import DatasetPoseCSV
from Tools.process import read_labels, rotate_img
from Tools.post_process import draw_relation
from GCN.graph_inference_block import graph_inference


def heatmap_to_points(heatmap, numPoints, ori_W, ori_H, resize_w, resize_h):
    keyPoints = np.zeros([numPoints, 2])  # 第一行为y（对应行数），第二行为x（对应列数）
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        # if max(max(row) for row in hm) < 0.1:
            # print('max:', max(max(row) for row in hm))
        center = np.unravel_index(np.argmax(hm), hm.shape)   # 寻找最大值点
        keyPoints[j, 0] = center[1] * ori_W / resize_w   # X
        keyPoints[j, 1] = center[0] * ori_H / resize_h  # Y
    return keyPoints

def predict_img(net,
                Img,
                device,
                resize_w,
                resize_h):
    net.eval()

    Img = DatasetPoseCSV.preprocess(resize_w, resize_h, Img, trans=1)
    Img = torch.from_numpy(Img)  # self.resize_w, self.resize_h, img0, self.scale, 1
    Img = Img.unsqueeze(0)
    Img = Img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output1 = net(Img)

        probs = F.softmax(output1, dim=1)
        probs = probs.squeeze(0)
        output1 = probs.cpu()

    return output1

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
    keyPoints = heatmap_to_points(heatmap, num_points, resize_w, resize_h, resize_w, resize_h)

    return keyPoints

def inference_PoseNet(net, GraphNet, device, csv_path, dir_img, out_files, num_points, resize_w, resize_h, pl_file, ul_file):
    # 读取测试数据列表
    labels = read_labels(csv_path)  # 读取标签
    imgs_num = len(labels)
    label_head = labels[0]

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

    isExists = os.path.exists(out_files + '/Right/')
    if not isExists:  # 创建结果文件夹
        # os.makedirs(out_files)
        os.makedirs(out_files + '/Right/')
        os.makedirs(out_files + '/False/')

    # 生成csv，写表头
    # pl_file = open(out_files + 'Right/Pesudo-label.csv', 'w+', newline='')
    keypointsWriter = csv.writer(pl_file)
    # ul_file = open(out_files + 'Right/noPesudo-label.csv', 'w+', newline='')
    keypointsWriter2 = csv.writer(ul_file)
    firstRow = label_head[:3 + num_points]
    keypointsWriter.writerow(firstRow)
    keypointsWriter2.writerow(firstRow)

    # for k in range(1, imgs_num):
    for k in range(1, 1000):
        # 读取图像
        img_path = labels[k][0]
        fn = os.path.join(dir_img + img_path)

        img0 = cv2.imread(fn)
        H, W, C = img0.shape
        # imgf = cv2.flip(img0, args.flip)  # 翻转 0表示上下，正数表示左右，负数表示上下左右都翻转
        # imgf = cv2.rotate(img0, cv2.ROTATE_180)
        imgf = rotate_img(img0, 180, [W / 2, H / 2])

        # 网络预测
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

        # 图推理
        pred = graph_inference(GraphNet, keyPoints0, keyPoints_f, num_points, device)

        searchContext1 = '/'
        numList = [m.start() for m in re.finditer(searchContext1, labels[k][0])]
        filename = img_path[numList[0] + 1:numList[1]] + '_' + img_path[numList[1] + 1:]

        if pred == 1:
            # print('keyPoints:', keyPoints0.shape, keyPoints0, keyPoints_f)
            # keyPoints0, keyPoints_f = refine_points(keyPoints0, keyPoints_f, resize_w, resize_h, args.flip)
            keypointsRow = [img_path, str(1),
                            str(int(min(keyPoints0[:, 0] * (W / resize_w)))) + '_' +
                            str(int(min(keyPoints0[:, 1] * (H / resize_h)))) + '_' +
                            str(int(max(keyPoints0[:, 0] * (W / resize_w)))) + '_' +
                            str(int(max(keyPoints0[:, 1] * (H / resize_h))))]
            keypointsRow_f = [img_path[:-4] + '_flip' + str(180) + '.jpg', str(1),
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

            save_name = out_files + '/Right/' + filename
            save_name_f = out_files + '/Right/' + filename[:-4] + '_f' + str(1) + '.jpg'
        else:
            keypointsRow = [img_path, str(1), '0_0_0_0']
            for p in range(num_points):
                keypointsRow.append('0_0_0')
            keypointsWriter2.writerow(keypointsRow)
            save_name = out_files + '/False/' + filename
            save_name_f = out_files + '/False/' + filename[:-4] + '_f0.jpg'

        # 保存图像
        if k % 100 == 0:
            print(k, img_path, save_name)

        #draw_relation(Img, allPoints, relations, resize_w, resize_h)
        img_0 = draw_relation(img0, keyPoints0, relation, resize_w, resize_h)
        img_f = draw_relation(imgf, keyPoints_f, relation, resize_w, resize_h)
        cv2.imwrite(save_name, img_0)
        cv2.imwrite(save_name_f, img_f)
