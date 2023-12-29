import cv2
import csv
import numpy as np


def read_labels(label_dir):
    # 读标签
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
    return labels


def rotate_img(image, angle, center):  # 对输入图像以点center为中心，旋转angle度
    # 构造变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 应用变换矩阵到图像
    rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated_img


def Combine_csv(file1_path, file2_path, new_path):
    labels1 = read_labels(file1_path)
    labels2 = read_labels(file2_path)

    # 生成新csv文件
    with open(new_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for l in range(len(labels1)):
            writer.writerow(labels1[l])
        for l in range(1, len(labels2)):
            writer.writerow(labels2[l])


def refine_points(p1, p2, width, height, flip_type):  # flip_type： 0表示上下，正数表示左右，负数表示上下左右都翻转
    numper_p = p1.shape[0]
    if flip_type == 0:
        temp = np.ones(numper_p) * height
        p2[:, 1] = temp - p2[:, 1]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
        mean_p = (p1 + p2) / 2
        p1 = mean_p
        p2[:, 0] = mean_p[:, 0]
        p2[:, 1] = temp - mean_p[:, 1]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
    elif flip_type > 0:
        temp = np.ones(numper_p) * width
        p2[:, 0] = temp - p2[:, 0]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
        mean_p = (p1 + p2) / 2
        p1 = mean_p
        p2[:, 0] = temp - mean_p[:, 0]
        p2[:, 1] = mean_p[:, 1]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
    else:
        temp = np.ones((numper_p, 2))
        temp[:, 0] = temp[:, 0] * width
        temp[:, 1] = temp[:, 1] * height
        p2_flip = temp - p2
        mean_p = (p1 + p2_flip) / 2
        p1 = mean_p
        p2 = temp - p2_flip
    return p1, p2


def cal_maxDis(p0):
    max_d = 0
    for i in range(p0.shape[0]):
        d = np.norm(p0[i, :])
        if d > max_d:
            max_d = d
    return max_d


def refine_points_v2(p1, p2, width, height, flip_type):  # flip_type： 0表示上下，正数表示左右，负数表示上下左右都翻转
    numper_p = p1.shape[0]
    if flip_type == 0:
        temp = np.ones(numper_p) * height
        p2[:, 1] = temp - p2[:, 1]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
        error_p = p1 - p2
        max_d = cal_maxDis(error_p)
    elif flip_type > 0:
        temp = np.ones(numper_p) * width
        p2[:, 0] = temp - p2[:, 0]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
        mean_p = (p1 + p2) / 2
        p1 = mean_p
        p2[:, 0] = temp - mean_p[:, 0]
        p2[:, 1] = mean_p[:, 1]
        # 左右肢交换
        t = np.copy(p2[0, :])
        p2[0, :] = p2[1, :]
        p2[1, :] = t
        t = np.copy(p2[2, :])
        p2[2, :] = p2[3, :]
        p2[3, :] = t
    else:
        temp = np.ones((numper_p, 2))
        temp[:, 0] = temp[:, 0] * width
        temp[:, 1] = temp[:, 1] * height
        p2_flip = temp - p2
        mean_p = (p1 + p2_flip) / 2
        p1 = mean_p
        p2 = temp - p2_flip
    return p1, p2