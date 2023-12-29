import cv2
import numpy as np


def cal_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[0]-p2[0])**2)


def heatmap_to_points(heatmap, numPoints, ori_W, ori_H, resize_w, resize_h):
    keyPoints = np.zeros([numPoints, 2])  # 第一行为y（对应行数），第二行为x（对应列数）
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        if max(max(row) for row in hm) < 0.1:
            print('max:', max(max(row) for row in hm))
        center = np.unravel_index(np.argmax(hm), hm.shape)   # 寻找最大值点
        keyPoints[j, 0] = center[1] * ori_W / resize_w   # X
        keyPoints[j, 1] = center[0] * ori_H / resize_h  # Y
    return keyPoints


def heatmap_to_points0(heatmap, numPoints, ori_W, ori_H):
    keyPoints = np.zeros([numPoints, 2])  # 第一行为y（对应行数），第二行为x（对应列数）
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        mx = np.max(hm)
        if mx > 0.2:
            center = np.unravel_index(np.argmax(hm), hm.shape)   # 寻找最大值点
            keyPoints[j, 0] = center[1]   # X
            keyPoints[j, 1] = center[0]   # Y
    return keyPoints


def draw_heatmap(Img, heatmap, numPoints, ori_W, ori_H, outfile, Filename, keypoints):
    Img = cv2.resize(Img, (ori_W, ori_H)) * 0.5
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        NormMap = (hm - np.mean(hm)) / (np.max(hm) - np.mean(hm))
        NormMap2 = (NormMap + np.abs(NormMap)) / 2
        map = np.round(NormMap2 * 100)
        Img[:, :, 1] = map + Img[:, :, 1]

        name = outfile + Filename + '_' + keypoints[j] + '.jpg'
        cv2.imwrite(name, Img)
    return 0


def draw_relation(Img, allPoints, relations, resize_w, resize_h):  # 640*480
    Img = cv2.resize(Img, (resize_w, resize_h))
    for k in range(len(relations)):
        c_x1 = int(allPoints[relations[k][0]-1, 0])
        c_y1 = int(allPoints[relations[k][0]-1, 1])
        c_x2 = int(allPoints[relations[k][1]-1, 0])
        c_y2 = int(allPoints[relations[k][1]-1, 1])
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    r = np.arange(50, 255, int(205/len(allPoints)))

    for j in range(len(allPoints)):  # 在原图中画出关键点
        cv2.circle(Img, (int(allPoints[j, 0]), int(allPoints[j, 1])), 2,
                   [int(r[len(allPoints) - j]), 20, int(r[j])], 2)
    return Img


def draw_relation2(Img, allPoints, relations):  # 640*480
    Img = cv2.resize(Img, (640, 480))
    for k in range(len(relations)):
        c_x1 = int(allPoints[relations[k][0]-1, 0] * (640/resize_w))
        c_y1 = int(allPoints[relations[k][0]-1, 1] * (480/resize_h))
        c_x2 = int(allPoints[relations[k][1]-1, 0] * (640/resize_w))
        c_y2 = int(allPoints[relations[k][1]-1, 1] * (480/resize_h))
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    r = np.arange(50, 255, int(205/len(allPoints)))

    for j in range(len(allPoints)):  # 在原图中画出关键点
        cv2.circle(Img, (int(allPoints[j, 0] * (640/resize_w)), int(allPoints[j, 1] * (480/resize_h))), 2,
                   [int(r[len(allPoints) - j]), 20, int(r[j])], 2)
    return Img
