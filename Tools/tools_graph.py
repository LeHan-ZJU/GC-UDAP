import cv2
import math
import random
import numpy as np
from Tools.process import rotate_img


def rotate_coordinates(p, angle, center):  # 计算点p绕点center旋转a度后的坐标
    angle_radians = math.radians(-angle)
    translated_x = p[0] - center[0]
    translated_y = p[1] - center[1]
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    rotated_x = cos_angle * translated_x - sin_angle * translated_y
    rotated_y = sin_angle * translated_x + cos_angle * translated_y
    final_x = rotated_x + center[0]
    final_y = rotated_y + center[1]
    return [final_x, final_y]


def aug_data(points, img, a_rotate):
    # 对输入图像进行随机增广
    # a_rotate = random.randint(0, 360)  # 随机生成一个0-360之间的旋转角
    # rot_points, rot_angle, rot_center = random_aug(points, a_rotate)  # 以目标中心点旋转

    h, w, _ = img.shape
    rot_center = [w / 2, h / 2]
    rot_points, rot_angle, rot_center = random_aug_v2(points, a_rotate, rot_center)    # 以图像中心点旋转
    rot_img = rotate_img(img, rot_angle, rot_center)
    return rot_points, rot_img


def random_aug(p1, a_rotate):
    # 以目标中心点旋转
    coor_r = np.zeros((2, p1.shape[1]))
    center = [0, 0]
    s = 0
    for i in range(p1.shape[1]):
        if min(p1[:, i]) > 0:
            center = center + p1[:, i]
            s = s + 1
    center = np.array(center)
    center = center / s

    for i in range(p1.shape[1]):
        if max(p1[:, i]) == 0:
            coor_r[:, i] = p1[:, i]
        else:
            coor_r[:, i] = rotate_coordinates(p1[:, i], a_rotate, center)

    return coor_r, a_rotate, center


def random_aug_v2(p1, a_rotate, center):
    # 以图像中心点旋转
    coor_r = np.zeros((2, p1.shape[1]))

    for i in range(p1.shape[1]):
        if max(p1[:, i]) == 0:
            coor_r[:, i] = p1[:, i]
        else:
            coor_r[:, i] = rotate_coordinates(p1[:, i], a_rotate, center)

    return coor_r, a_rotate, center


def find_length(p1, p2):  # 计算两点x、y之间的距离，并返回较大的一个
    l1 = int(np.abs(p1[0] - p2[0]))
    l2 = int(np.abs(p1[1] - p2[1]))
    return max(l1, l2)


def find_neighbor(sort, edge_index):
    neighbor = []
    size_edge = edge_index.shape
    for n in range(size_edge[1]):
        if edge_index[0, n] == sort:
            neighbor.append(edge_index[1, n])
    return np.array(neighbor)


def cal_Angle(v1, v2):
    # 计算两个向量间的夹角
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    # norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
    # norm2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
    m = np.dot(v1, v2)
    cos_ = m / (norm1 * norm2)
    inv = np.arccos(cos_) * 180 / np.pi

    if np.isnan(inv):
        inv = 0

    return inv


def show_img(Img, allPoints, num_points):
    allPoints = allPoints[:, 0:2]
    if num_points == 6:
        relations = [[0, 3], [0, 4], [1, 2], [1, 4], [2, 5], [3, 5]]   # rat
    else:
        relations = [[0, 1], [0, 2], [0, 12], [0, 18], [1, 2], [1, 13], [1, 18], [2, 13], [2, 18], [3, 4], [3, 6], [3, 8],
                 [3, 10], [4, 5], [4, 6], [4, 9], [5, 6], [5, 10], [6, 11], [7, 12], [7, 13], [7, 16], [7, 17], [8, 9],
                 [8, 10], [8, 14], [9, 11], [9, 15], [10, 11], [10, 16], [11, 17], [12, 14], [12, 18], [13, 15],
                 [13, 18], [14, 15], [14, 16], [15, 17], [16, 17]]  # TigDog

    # print('relations:', len(relations))
    for k in range(len(relations)):
        # print(k, relations[k][0], relations[k][1], allPoints.shape)
        if max(allPoints[relations[k][0], :]) > 0 and max(allPoints[relations[k][1], :]) > 0:
            c_x1 = int(allPoints[relations[k][0], 0])
            c_y1 = int(allPoints[relations[k][0], 1])
            c_x2 = int(allPoints[relations[k][1], 0])
            c_y2 = int(allPoints[relations[k][1], 1])
            cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)

    r = np.arange(50, 255, int(205 / len(relations)))

    # for j in range(max(max(relations)) + 2):  # 在原图中画出关键点
    for j in range(num_points):
        cv2.circle(Img, (int(allPoints[j, 0]), int(allPoints[j, 1])), 2,
                   [int(r[len(relations) - j]), 20, int(r[j])], 2)
    return Img


def position_consistency(p1, p2, angle, w, h, thr=50):
    if angle == 180:
        temp_w = np.ones(p1.shape[0]) * w
        temp_h = np.ones(p1.shape[0]) * h
        t = np.copy(p2)
        t[:, 0] = temp_w - t[:, 0]
        t[:, 1] = temp_h - t[:, 1]
        p2 = t

    s = 0
    for i in range(p1.shape[0]):
        d = math.sqrt((p1[i, 0] - p2[i, 0])**2 + (p1[i, 1] - p2[i, 1])**2)
        if d > thr:
            s = s + 1
    return s
