"""
# --------------------------------------------------------
# @Project: 图连接对分类网络的数据读取
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2023-05-29
# --------------------------------------------------------
"""
import re
import cv2
import csv
import torch
import random
import numpy as np
from PIL import Image
from torch_geometric.data import Data
from Tools.process import rotate_img
from Tools.tools_graph import random_aug, find_length, find_neighbor, cal_Angle, show_img, aug_data


def read_edge_index(num_points, s=0):
    if num_points == 6:
        if s == 0:  # 非孪生网络结构
            edge_index = [[0, 3], [3, 0], [0, 4], [4, 0],   # 2 neighbor
                          [1, 2], [2, 1], [1, 4], [4, 1],
                          [2, 5], [5, 2], [3, 5], [5, 3],
                          [6, 9], [9, 6], [6, 10], [10, 6],
                          [7, 8], [8, 7], [7, 10], [10, 7],
                          [8, 11], [11, 8], [9, 11], [11, 9]]
            # edge_index = [[0, 3], [3, 0], [0, 4], [4, 0],     # 3 neighbor
            #               [1, 2], [2, 1], [1, 4], [4, 1],
            #               [2, 5], [5, 2], [3, 5], [5, 3],
            #               [0, 2], [2, 0], [1, 3], [3, 1], [4, 5], [5, 4],
            #               [6, 9], [9, 6], [6, 10], [10, 6],
            #               [7, 8], [8, 7], [7, 10], [10, 7],
            #               [8, 11], [11, 8], [9, 11], [11, 9],
            #               [6, 8], [8, 6], [7, 9], [9, 7], [10, 11], [10, 11]]
    elif num_points == 10:
        if s == 0:  # 非孪生网络结构
            edge_index = [[0, 2], [2, 0], [0, 5], [5, 0], [0, 7], [7, 0],
                          [1, 3], [3, 1], [1, 5], [5, 1], [1, 7], [7, 1],
                          [2, 4], [4, 2], [2, 6], [6, 2],
                          [3, 4], [4, 3], [3, 6], [6, 3],
                          [5, 7], [7, 5], [5, 8], [8, 5],
                          [6, 8], [8, 6], [8, 9], [9, 8],
                          [10, 12], [12, 10], [10, 15], [15, 10], [10, 17], [17, 10],
                          [11, 13], [13, 11], [11, 15], [15, 11], [11, 17], [17, 11],
                          [12, 14], [14, 12], [12, 16], [16, 12],
                          [13, 14], [14, 13], [13, 16], [16, 13],
                          [15, 17], [17, 15], [15, 18], [18, 15],
                          [16, 18], [18, 16], [18, 19], [19, 18]]
    elif num_points == 19:
        if s == 0:
            edge_index = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 12], [12, 0], [0, 18], [18, 0],
                          [1, 2], [2, 1], [1, 13], [13, 1], [1, 18], [18, 1],
                          [2, 13], [13, 2], [2, 18], [18, 2],
                          [3, 4], [4, 3], [3, 5], [5, 3], [3, 6], [6, 3], [3, 8], [8, 3],
                          [4, 5], [5, 4], [4, 6], [6, 4], [4, 9], [9, 4],
                          [5, 6], [6, 5], [5, 10], [10, 5], [6, 11], [11, 6],
                          [7, 12], [12, 7], [7, 13], [13, 7], [7, 16], [16, 7], [7, 17], [17, 7],
                          [8, 9], [9, 8], [8, 10], [10, 8], [8, 14], [14, 8],
                          [9, 11], [11, 9], [9, 15], [15, 9],
                          [10, 11], [11, 10], [10, 16], [16, 10], [11, 17], [17, 11],
                          [12, 14], [14, 12], [12, 18], [18, 12],
                          [13, 15], [15, 13],
                          [14, 15], [15, 14], [14, 16], [16, 14],
                          [15, 17], [17, 15], [16, 17], [17, 16],
                          [19, 20], [20, 19], [19, 21], [21, 19], [19, 31], [31, 19], [19, 37], [37, 19],
                          [20, 21], [21, 20], [20, 32], [32, 20], [20, 37], [37, 20],
                          [21, 32], [32, 21], [21, 37], [37, 21],
                          [22, 23], [23, 22], [22, 24], [24, 22], [22, 25], [25, 22], [22, 27], [27, 22],
                          [23, 24], [24, 23], [23, 25], [25, 23], [23, 28], [28, 23],
                          [24, 25], [25, 24], [24, 29], [29, 24], [25, 30], [30, 25],
                          [26, 31], [31, 26], [26, 32], [32, 26], [26, 35], [35, 26], [26, 36], [36, 26],
                          [27, 28], [28, 27], [27, 29], [29, 27], [27, 33], [33, 27],
                          [28, 30], [30, 28], [28, 34], [34, 28],
                          [29, 30], [30, 29], [29, 35], [35, 29], [30, 36], [36, 30],
                          [31, 33], [33, 31], [31, 37], [37, 31],
                          [32, 34], [34, 32],
                          [33, 34], [34, 33], [33, 35], [35, 33],
                          [34, 36], [36, 34], [35, 36], [36, 35],
                          ]

    else:
        edge_index = []

    return np.array(edge_index)


def Angle_relation_multiAngles(AllPoints, edge_index, num_points):
    angles = []
    for p in range(num_points):
        if num_points == 6:
            angle = np.ones(1)
        else:
            angle = np.ones(6)
        point0 = AllPoints[0:2, p]
        if min(point0) > 0:
            # 根据edge_index寻找与它相邻的两个点
            neighbors = find_neighbor(p, edge_index)
            # print('neighbors:', p, neighbors, len(neighbors))
            if len(neighbors) >= 2:
                s = 0
                for v1 in range(len(neighbors) - 1):
                    for v2 in range(v1 + 1, len(neighbors)):
                        if max(AllPoints[0:2, neighbors[v1]]) == 0 and min(AllPoints[0:2, neighbors[v2]]) > 0:
                            angle[s] = 0  # p非空时，v1和v2任一为[0, 0] --> a=0
                        elif max(AllPoints[0:2, neighbors[v2]]) == 0 and min(AllPoints[0:2, neighbors[v1]]) > 0:
                            angle[s] = 0
                        elif max(AllPoints[0:2, neighbors[v1]]) == 0 and max(AllPoints[0:2, neighbors[v2]]) == 0:
                            angle[s] = -1    # p非空时，v1和v2均为[0, 0] --> a=-1
                        else:
                            # if min(AllPoints[:, neighbors[v1]]) > 0 and min(AllPoints[:, neighbors[v2]]) > 0:
                            vector1 = AllPoints[0:2, neighbors[v1]] - point0
                            vector2 = AllPoints[0:2, neighbors[v2]] - point0
                            angle[s] = cal_Angle(vector1, vector2)
                        s = s + 1
        else:
            angle = -2 * angle  # p:[0, 0] --> a=-2
        angle = np.array(angle)
        # angle = angle[0]

        angles.append(angle)
    angles = np.array(angles)
    # angles = angles.astype(np.float32)

    return angles


def add_noise(points, neg, length, w, h, noise_n, noise_p):  # neg=0:正样本对；1：一正一负；2：负样本对
    n_points = points.shape[1]
    if neg == 0:  # 构造正样本
        for n in range(n_points):
            if max(points[:, n]) == 0:
                points_noise = points[:, n]
            else:
                points_noise = [
                    min(max(points[0, n] + random.randint(int(-length * noise_p), int(length * noise_p)), 0), w),
                    min(max(points[1, n] + random.randint(int(-length * noise_p), int(length * noise_p)), 0), h)
                ]
            points[0:2, n] = points_noise
    else:  # 构造负样本
        for n in range(n_points):
            if max(points[:, n]) == 0:
                points_noise = points[:, n]
            else:
                # 生成大噪声
                noise = [
                    random.randint(int(length * 0), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0]),
                    random.randint(int(length * 0), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0])
                ]
                points_noise = [
                    min(max(points[0, n] + int(noise[0]), 0), w),
                    min(max(points[1, n] + int(noise[1]), 0), h)
                ]
            points[0:2, n] = points_noise
    return points


def read_data1(labels, imgs_dir, num_points, index1, noise):
    # 读取第i个样本标签
    img_name = imgs_dir + labels[index1][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    numList = [m.start() for m in re.finditer("_", labels[index1][2])]  # 读取box标签获得目标长度，以确定扰动范围
    box = [int(labels[index1][2][0:numList[0]]), int(labels[index1][2][numList[0] + 1:numList[1]]),
           int(labels[index1][2][numList[1] + 1:numList[2]]), int(labels[index1][2][numList[2] + 1:])]
    length = find_length(box[0:2], box[2:4])

    points_all = np.zeros([8, num_points * 2])  # 存放当前图中的所有人的所有关键点
    for n in range(num_points):
        numList1 = [m.start() for m in re.finditer("_", labels[index1][n + 3])]
        p_x = int(labels[index1][n + 3][0:numList1[0]])
        p_y = int(labels[index1][n + 3][numList1[0] + 1:numList1[1]])
        if max([p_x, p_y]) == 0:
            point = [p_x, p_y]
            # point = [-1, -1]
        else:
            point = [min(max(p_x + random.randint(int(-length * noise), int(length * noise)), 0), w),
                     min(max(p_y + random.randint(int(-length * noise), int(length * noise)), 0), h)]
        points_all[0:2, n] = point
    show = show_img(img, points_all[:, 0:num_points].T, num_points)

    return points_all, length, show


def gen_data1(labels, imgs_dir, num_points, index1, negative, noise_p, noise_n):
    # 读取第i个样本标签
    img_name = imgs_dir + labels[index1][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    numList = [m.start() for m in re.finditer("_", labels[index1][2])]  # 读取box标签获得目标长度，以确定扰动范围
    box = [int(labels[index1][2][0:numList[0]]), int(labels[index1][2][numList[0] + 1:numList[1]]),
           int(labels[index1][2][numList[1] + 1:numList[2]]), int(labels[index1][2][numList[2] + 1:])]
    length = find_length(box[0:2], box[2:4])

    points_all = np.zeros([8, num_points * 2])  # 存放当前图中的所有人的所有关键点
    # 读points1的时候，原始关键点存放到point_all的后半段，前半段放points加噪后的值
    for n in range(num_points):
        numList1 = [m.start() for m in re.finditer("_", labels[index1][n + 3])]
        points_all[0, n + num_points] = int(labels[index1][n + 3][0:numList1[0]])
        points_all[1, n + num_points] = int(labels[index1][n + 3][numList1[0] + 1:numList1[1]])
    points_all[0:2, 0:num_points] = points_all[0:2, num_points:]

    if negative[0] == 0:
        neg = 1
    elif negative[0] == 1 and negative[1] == 1:
        neg = 1
    else:
        neg = 0
    points_all[0:2, 0:num_points] = add_noise(points_all[0:2, 0:num_points], neg, length, w, h, noise_n, noise_p)

    # show = show_img(img, points_all[:, 0:num_points].T, num_points)

    return points_all, length  # , show


def read_data2(labels, imgs_dir, num_points, index2, length, points_all, negative, noise_p, noise_n):
    img_name = imgs_dir + labels[index2][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    if negative == 0:  # 构造正样本
        y = 1
        for n in range(num_points):
            numList2 = [m.start() for m in re.finditer("_", labels[index2][n + 3])]
            point2 = [min(max(int(labels[index2][n + 3][0:numList2[0]]) +
                          random.randint(int(-length * noise_p), int(length * noise_p)), 0), w),
                      min(max(int(labels[index2][n + 3][numList2[0] + 1:numList2[1]]) +
                          random.randint(int(-length * noise_p), int(length * noise_p)), 0), h)]
            points_all[0:2, n + num_points] = point2
    else:  # 构造负样本
        y = 0
        for n in range(num_points):
            numList2 = [m.start() for m in re.finditer("_", labels[index2][n + 3])]
            # 生成大噪声
            # noise = [random.randint(int(length * 0.05),  int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0]),
            #         random.randint(int(length * 0.05), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0])]
            noise = [
                random.randint(int(length * 0), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0]),
                random.randint(int(length * 0), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0])]

            point2 = [min(max(int(labels[index2][n + 3][0:numList2[0]]) + int(noise[0]), 0), w),
                      min(max(int(labels[index2][n + 3][numList2[0] + 1:numList2[1]]) + int(noise[1]), 0), h)]
            points_all[0:2, n + num_points] = point2
    show = show_img(img, points_all[:, 6:].T)

    return points_all, np.array(y), show


def gen_data2(labels, imgs_dir, num_points, index2, length, points_all, negative, noise_p, noise_n):
    img_name = imgs_dir + labels[index2][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    # 对输入图像进行随机增广
    ini_points = points_all[0:2, 0:num_points]
    rot_points, rot_angle, rot_center = random_aug(ini_points)
    rot_img = rotate_img(img, rot_angle, rot_center)

    if negative == 0:  # 构造正样本
        y = 1
        for n in range(num_points):
            if max(rot_points[:, n]) == 0:
                point2 = rot_points[:, n]
                # point2 = [-1, -1]
            else:
                # point2 = rot_points[:, n]
                point2 = [
                    min(max(rot_points[0, n] + random.randint(int(-length * noise_p), int(length * noise_p)), 0), w),
                    min(max(rot_points[1, n] + random.randint(int(-length * noise_p), int(length * noise_p)), 0), h)
                ]
            points_all[0:2, n + num_points] = point2
    else:  # 构造负样本
        y = 0
        for n in range(num_points):
            if max(rot_points[:, n]) == 0:
                point2 = rot_points[:, n]
            else:
                # 生成大噪声
                noise = [
                    random.randint(int(length * 0), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0]),
                    random.randint(int(length * 0), int(length * noise_n)) * ((-1) ** random.sample(range(2), 1)[0])
                ]
                point2 = [
                    min(max(rot_points[0, n] + int(noise[0]), 0), w),
                    min(max(rot_points[1, n] + int(noise[1]), 0), h)
                ]
            points_all[0:2, n + num_points] = point2
    show = show_img(rot_img, points_all[:, num_points:].T, num_points)

    return points_all, np.array(y), show


def read_data_resize(labels, imgs_dir, num_points, index1, re_w, re_h):
    # 读取第i个样本标签
    img_name = imgs_dir + labels[index1][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (re_w, re_h))
    h, w, _ = img.shape

    numList = [m.start() for m in re.finditer("_", labels[index1][2])]  # 读取box标签获得目标长度，以确定扰动范围
    box = [int(labels[index1][2][0:numList[0]]) * (re_w / w),
           int(labels[index1][2][numList[0] + 1:numList[1]]) * (re_h / h),
           int(labels[index1][2][numList[1] + 1:numList[2]]) * (re_w / w),
           int(labels[index1][2][numList[2] + 1:]) * (re_h / h)]
    length = find_length(box[0:2], box[2:4])

    points_all = np.zeros([2, num_points])  # 存放当前图中的所有人的所有关键点
    # 读points
    for n in range(num_points):
        numList1 = [m.start() for m in re.finditer("_", labels[index1][n + 3])]
        points_all[0, n] = int(labels[index1][n + 3][0:numList1[0]]) * (re_w / w)
        points_all[1, n] = int(labels[index1][n + 3][numList1[0] + 1:numList1[1]]) * (re_h / h)

    return h, w, length, points_all, img


def read_data(labels, imgs_dir, num_points, index1):
    # 读取第i个样本标签
    img_name = imgs_dir + labels[index1][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    numList = [m.start() for m in re.finditer("_", labels[index1][2])]  # 读取box标签获得目标长度，以确定扰动范围
    box = [int(labels[index1][2][0:numList[0]]),
           int(labels[index1][2][numList[0] + 1:numList[1]]),
           int(labels[index1][2][numList[1] + 1:numList[2]]),
           int(labels[index1][2][numList[2] + 1:])]
    length = find_length(box[0:2], box[2:4])

    points_all = np.zeros([2, num_points])  # 存放当前图中的所有人的所有关键点
    # 读points
    for n in range(num_points):
        numList1 = [m.start() for m in re.finditer("_", labels[index1][n + 3])]
        points_all[0, n] = int(labels[index1][n + 3][0:numList1[0]])
        points_all[1, n] = int(labels[index1][n + 3][numList1[0] + 1:numList1[1]])

    return h, w, length, points_all, img


def gen_sample_sNoise(points, w, h, length, rate_p):
    n_points = points.shape[1]
    for n in range(n_points):
        if max(points[:, n]) == 0:
            points_noise = points[:, n]
        else:
            points_noise = [
                min(max(points[0, n] + random.randint(int(-length * rate_p), int(length * rate_p)), 0), w),
                min(max(points[1, n] + random.randint(int(-length * rate_p), int(length * rate_p)), 0), h)
            ]
        points[:, n] = points_noise
    return points


def gen_sample_bNoise(points, w, h, length, rate_p, rate_n):
    n_points = points.shape[1]
    for n in range(n_points):
        if max(points[:, n]) == 0:
            points_noise = points[:, n]
        else:
            # 1/3的概率生成大噪声
            ra = random.sample(range(2), 1)[0]
            if ra == 0:
                noise = [
                    random.randint(int(length * 0), int(length * rate_n)) * ((-1) ** random.sample(range(2), 1)[0]),
                    random.randint(int(length * 0), int(length * rate_n)) * ((-1) ** random.sample(range(2), 1)[0])]
            else:
                noise = [random.randint(int(-length * rate_p), int(length * rate_p)),
                         random.randint(int(-length * rate_p), int(length * rate_p))]

            points_noise = [
                min(max(points[0, n] + int(noise[0]), 0), w),
                min(max(points[1, n] + int(noise[1]), 0), h)
            ]
        points[:, n] = points_noise
    return points


def gen_sample_bNoise_same(P1, P2, w, h, length, rate_p, rate_n):
    n_points = P1.shape[1]

    for n in range(n_points):
        # 1/3的概率生成大噪声
        ra = random.sample(range(2), 1)[0]
        if ra == 0:
            noise = [random.randint(int(length * 0), int(length * rate_n)) * ((-1) ** random.sample(range(2), 1)[0]),
                     random.randint(int(length * 0), int(length * rate_n)) * ((-1) ** random.sample(range(2), 1)[0])]
        else:
            noise = [random.randint(int(-length * rate_p), int(length * rate_p)),
                     random.randint(int(-length * rate_p), int(length * rate_p))]

        if max(P1[:, n]) == 0:  # P1加噪
            points_noise1 = P1[:, n]
        else:
            points_noise1 = [min(max(P1[0, n] + int(noise[0]), 0), w),
                             min(max(P1[1, n] + int(noise[1]), 0), h)]

        if max(P2[:, n]) == 0:  # P2加噪
            points_noise2 = P2[:, n]
        else:
            points_noise2 = [min(max(P2[0, n] + int(noise[0]), 0), w),
                             min(max(P2[1, n] + int(noise[1]), 0), h)]

        P1[:, n] = points_noise1
        P2[:, n] = points_noise2
    return P1, P2


def gcn_dataset_tigdog(dir_label, dir_img, num_points, noise_p, noise_n):
    data_list =[]
    # 读csv文件
    with open(dir_label, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        labels = list(reader)
        print('len', len(labels))
        trainset_num = len(labels)

    # 挨个读取关键点，生成训练数据
    neg_percent = 0
    for i in range(1, trainset_num):
        index1 = i

        RandomRate = random.sample(range(3), 1)  # 百分之五十的概率构造正负样本
        negative = 1
        if RandomRate[0] == 0:  # 构造负样本：百分之五十为对原数据标签加较大噪声，百分之五十为错配样本标签
            negative = 0

        edge_index = read_edge_index(num_points, 0)
        x, length, show1 = read_data1(labels, dir_img, num_points, index1, noise=noise_p)
        angles1 = Angle_relation_multiAngles(x[:, 0:num_points], edge_index.T, num_points)
        # x, y = read_data2(labels, dir_img, num_points, index2, length, x, negative, noise_p=noise_p, noise_n=noise_n)
        x, y, show2 = gen_data2(labels, dir_img, num_points, index1, length, x, negative, noise_p=noise_p, noise_n=noise_n)
        angles2 = Angle_relation_multiAngles(x[:, num_points:], edge_index.T, num_points)

        x[2:, 0:num_points] = angles1.T
        x[2:, num_points:] = angles2.T

        data = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor(y)

        Data_i = Data(x=data.t(), edge_index=edge_index.t(), y=y)
        data_list.append(Data_i)

        neg_percent = neg_percent + negative

    print('negative percent:', neg_percent / trainset_num)

    return data_list


def gcn_dataset_rat(dir_label, dir_img, num_points, noise_p, noise_n):
    data_list =[]
    # 读csv文件
    with open(dir_label, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        labels = list(reader)
        print('len', len(labels))
        trainset_num = len(labels)

    # 挨个读取关键点，生成训练数据
    for i in range(1, trainset_num):
        index1 = i
        x = np.zeros([8, num_points * 2])  # 初始化数据

        # 读取边连接顺序以及图像标签数据
        edge_index = read_edge_index(num_points)
        height, width, length, points1, img1 = read_data(labels, dir_img, num_points, index1)
        points1 = gen_sample_sNoise(points1, width, height, length, rate_p=noise_p)

        points2, img2 = aug_data(points1, img1, 180)

        postive = random.sample(range(3), 1)[0]  # 1:2的比例构造正负样本
        if postive == 1:
            points2 = gen_sample_sNoise(points2, width, height, length, rate_p=noise_p)

        else:
            postive = 0
            case = random.sample(range(3), 1)[0]
            if case == 0:    # case=0:p1p2加同样的噪声构建hard负样本对
                points1 = gen_sample_bNoise(points1, width, height, length, rate_p=noise_p, rate_n=noise_n)
                points2, img2 = aug_data(points1, img1, 180)
                # points1, points2 = gen_sample_bNoise_same(points1, points2, width, height, length, rate_p=noise_p, rate_n=noise_n)
            elif case == 1:  # case=1：p1 p2加不同大噪声
                points1 = gen_sample_bNoise(points1, width, height, length, rate_p=noise_p, rate_n=noise_n)
                points2 = gen_sample_bNoise(points2, width, height, length, rate_p=noise_p, rate_n=noise_n)
            else:            # case=2：随机一正一负
                r = random.sample(range(2), 1)[0]
                if r == 0:
                    points2 = gen_sample_bNoise(points2, width, height, length, rate_p=noise_p, rate_n=noise_n)
                else:
                    points1 = gen_sample_bNoise(points1, width, height, length, rate_p=noise_p, rate_n=noise_n)
                    points2 = gen_sample_sNoise(points2, width, height, length, rate_p=noise_p)


        x[0:2, 0:num_points] = points1
        x[0:2, num_points:] = points2
        angles1 = Angle_relation_multiAngles(points1, edge_index.T, num_points)
        angles2 = Angle_relation_multiAngles(points2, edge_index.T, num_points)
        x[2:, 0:num_points] = angles1.T
        x[2:, num_points:] = angles2.T

        data = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        postive = torch.tensor(postive)

        Data_i = Data(x=data.t(), edge_index=edge_index.t(), y=postive)
        data_list.append(Data_i)

    return data_list
