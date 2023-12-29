# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: MyPoseNet
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2021-06-17
# --------------------------------------------------------
"""

import csv
import os
import re
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from Tools.process import rotate_img
from Tools.tools_graph import rotate_coordinates
import logging
from PIL import Image

from Tools.process import rotate_img


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):  # 根据关键点的坐标生成heatmap
    if max(c_x, c_y) == 0:
        return np.zeros([img_width, img_height])
    else:
        X1 = np.linspace(1, img_width, img_width)
        Y1 = np.linspace(1, img_height, img_height)
        [X, Y] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap * 255
        return heatmap


def CenterLabelHeatMapResize(img_height, img_width, c_x, c_y, resize_h, resize_w, sigma):   # 根据关键点的坐标生成heatmap
    if max(c_x, c_y) == 0:
        return np.zeros([resize_h, resize_w])
    else:
        c_x = int(c_x * (resize_w / img_width))
        c_y = int(c_y * (resize_h / img_height))
        # sigma = max(int(sigma * (resize_h / img_height)), 1)

        Y1 = np.linspace(1, resize_w, resize_w)
        X1 = np.linspace(1, resize_h, resize_h)
        [X, Y] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap * 255
        return heatmap


def object_area(num, points_all):
    # fuction: 根据输入的一组关键点集合，先挑选出其中所有不全为0的点，然后两两组合

    # 选出所有不为0的点
    points = []
    # print(points_all)
    for sort in range(num):
        if min(points_all[sort, :]) > 0:
            points.append(points_all[sort, :])
    points = np.array(points)
    # print(points)

    # 建立连接关系
    relation = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            relation.append([i, j])
    relation = np.array(relation)
    # print(relation)

    area = []
    for i in range(len(relation)):
        area.append(points[relation[i, 0], :])
        area.append(points[relation[i, 1], :])

    return np.array(area)


def LocationMapResize(img_height, img_width, points_all, resize_h, resize_w, coco):   # 根据关键点的坐标生成heatmap
    num = points_all.shape
    locationmap = np.zeros([int(resize_h), int(resize_w), 3])  # 初始化定位标签
    locationmap = locationmap.astype(np.uint8)
    Points = []
    for i in range(num[1]):
        c_x = points_all[0, i, 0]
        c_y = points_all[0, i, 1]
        # if min(c_x, c_y) > 0:
        c_x = int(c_x * (resize_w / img_width))
        c_y = int(c_y * (resize_h / img_height))
        Points.append([c_x, c_y])
    Points = np.array(Points)

    if coco == 1:
        # area = find_edgePoints(num[1], Points)
        area = object_area(num[1], Points)
        area = area.astype(int)
        if len(area) > 0:
            cv2.fillConvexPoly(locationmap, area, (255, 255, 255))
            locationmap = cv2.GaussianBlur(locationmap[:, :, 0], (45, 45), 25)  # kernel_size=(45, 45)， sigma=45
        else:
            locationmap = locationmap[:, :, 0]

    else:
        area = np.array([Points[0, :], Points[2, :], Points[5, :], Points[3, :], Points[1, :], Points[4, :]])
        area = area.astype(int)
        cv2.fillConvexPoly(locationmap, area, (255, 255, 255))
        # locationmap = locationmap[:, :, 0]
        locationmap = cv2.GaussianBlur(locationmap[:, :, 0], (15, 15), 15)  # kernel_size=(45, 45)， sigma=45

    locationmap = cv2.resize(locationmap, (int(resize_w / 16), int(resize_h / 16)))
    locationmap = np.expand_dims(locationmap, axis=0)
    locationmap = np.array(locationmap / 255)

    # location_vec = locationmap.reshape(1, int(resize_w / 32) * int(resize_h / 32))
    # location_vec = np.array(location_vec.squeeze(0))
    # print('3:', location_vec.shape)

    return locationmap  # , location_vec


def box2map(Img):  # https://blog.csdn.net/qq_30154571/article/details/109557559
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)  # 灰度化

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 二值化

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 去噪

    dist_transform = cv2.distanceTransform(opening, 1, 5)  # 转为热力图
    # blur = cv2.blur(dist_transform, (5, 5))  # 模糊化
    # alpha = 255 / (blur.max())
    return dist_transform


def box2patch(Image, Annotation, resizeW, resizeH):
    # 裁剪box内部分
    H, W, C = Image.shape
    Image = cv2.resize(Image, (resizeW, resizeH))
    temp = np.zeros([resizeH, resizeW])

    numList = [m.start() for m in re.finditer('_', Annotation)]
    lu_x0 = int(float(Annotation[0:numList[0]]) * (resizeW / W))
    lu_y0 = int(float(Annotation[numList[0] + 1:numList[1]]) * (resizeH / H))
    rb_x0 = int(float(Annotation[numList[1] + 1:numList[2]]) * (resizeW / W))
    rb_y0 = int(float(Annotation[numList[2] + 1:]) * (resizeH / H))
    # print('box:', Annotation)

    # 坐标变换
    lu_x0, lu_y0 = Trans_point([lu_x0, lu_y0], H, W, [resizeH, resizeW])
    rb_x0, rb_y0 = Trans_point([rb_x0, rb_y0], H, W, [resizeH, resizeW])

    patch = Image[min(lu_y0, rb_y0):max(lu_y0, rb_y0), min(lu_x0, rb_x0):max(lu_x0, rb_x0), :]
    temp[min(lu_y0, rb_y0):max(lu_y0, rb_y0), min(lu_x0, rb_x0):max(lu_x0, rb_x0)] = box2map(patch)

    blur = cv2.blur(temp, (3, 3))  # 模糊化
    alpha = 255 / (blur.max())
    blur = blur * alpha
    # cv2.imshow('img_new', Image)
    # cv2.imshow('img_new2', np.uint8(blur))
    # cv2.waitKey(0)
    blur = cv2.resize(blur, (int(resizeW / 4), int(resizeH / 4)))
    blur = np.expand_dims(blur, axis=0)

    return blur


def AugImg(Img, resize):
    # 对图像按目标尺寸进行resize和padding，并且将标签也进行对应变换 resize = [H, W]
    h, w, _ = Img.shape
    wr = w * (resize[0] / h)    # 若按h进行resize，则得到的宽度应该为wr
    # hr = h * (resize[1] / w)  # 若按w进行resize，则得到的高度应该为hr
    # if max([h, w]) == w:  # 按w对齐
    if wr > resize[1]:
    # if max([h, w]) == w:   # 按w对齐
        res_2 = int(h * (resize[1] / w))
        Img = cv2.resize(Img, (resize[1], res_2))
        padding_l = int((resize[0] - res_2) / 2)
        if padding_l > 0:
            Img = cv2.copyMakeBorder(Img, padding_l, padding_l, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # 上下填充0

    else:   # 按h对齐
        res_2 = int(w * (resize[0] / h))
        Img = cv2.resize(Img, (res_2, resize[0]))
        padding_l = int((resize[1] - res_2) / 2)
        if padding_l > 0:
            Img = cv2.copyMakeBorder(Img, 0, 0, padding_l, padding_l, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # 左右填充0

    return Img


def Trans_point(p, h, w, resize):
    if min(p) > 0:
        wr = w * (resize[0] / h)  # 若按h进行resize，则得到的宽度应该为wr
        # hr = h * (resize[1] / w)  # 若按w进行resize，则得到的高度应该为hr
        # if max([h, w]) == w:  # 按w对齐
        if wr > resize[1]:
            res_2 = int(h * (resize[1] / w))
            padding_l = int((resize[0] - res_2) / 2)
            p[0] = int(p[0] * (resize[1] / w))  # x
            p[1] = int(p[1] * (resize[1] / w) + padding_l)  # y
        else:  # 按h对齐
            res_2 = int(w * (resize[0] / h))
            padding_l = int((resize[1] - res_2) / 2)
            p[0] = int(p[0] * (resize[0] / h) + padding_l)  # x
            p[1] = int(p[1] * (resize[0] / h))  # y
    return p


class DatasetPoseCSV(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            Point_name = self.labels[0][3:]
            # print('len', len(self.labels))
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                # print(self.labels[img_sort])
                self.name_list.append(img_sort)
                points_num = int(self.labels[img_sort][1])  # 当前图像中的人数
                # img_sort = img_sort+points_num
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        # print(img_file)
        # img0 = cv2.imread(img_file)
        img = Image.open(img_file)
        RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
        if RandomRate[0] == 1:
            img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
            img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)
        # index = self.labels[idx][0].index('.')
        searchContext = "_"
        # boxList = [m.start() for m in re.finditer(searchContext, self.labels[idx][2])]  # 获取第k个人Bounding box标签
        # box = [float(self.labels[idx][2][0:boxList[0]]), float(self.labels[idx][2][boxList[0] + 1:boxList[1]]),
        #        float(self.labels[idx][2][boxList[1] + 1:boxList[2]]), float(self.labels[idx][2][boxList[2] + 1:])]
        # w_box, h_box = int(box[2]), int(box[3])
        # 获取图像中第k个人的17个Key points
        # heatmaps = np.zeros([h, w, self.num_points7])
        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point
                # print(point)
                # sigma = max(int(h_box / 100), 1)  # 根据图中每个人的box高度确定高斯核参数sigma
                # sigma = max(int(w_box / 50), 1)
                sigma = 3

                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # # print(img.shape)
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.waitKey(0)


        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps = np.array(heatmaps)
        # print('img:', img.shape)
        # print('heatmaps:', heatmaps.shape)
        # print(img.shape)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  #,
        }


class DatasetPoseCSVLocation(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points, Aug, coco):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points
        self.Aug = Aug
        self.coco = coco

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)

            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:
                self.name_list.append(img_sort)
                points_num = int(self.labels[img_sort][1])  # 当前图像中的人数
                img_sort = img_sort+points_num
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, scale, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图

        img = Image.open(img_file)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', cv2.resize(img0, (self.resize_w, self.resize_h)))
        # cv2.waitKey(0)
        h, w, _ = img0.shape
        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img, self.scale, 1)
        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        # locationmap = np.zeros([int(self.resize_h), int(self.resize_w), 3])  # 初始化定位标签
        # locationmap = locationmap.astype(np.uint8)
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point
                sigma = 2
                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

        # area = np.array([points_all[0, 0, :], points_all[0, 2, :], points_all[0, 5, :], points_all[0, 3, :],
        #                      points_all[0, 1, :], points_all[0, 4, :]])
        # area = area.astype(int)
        # cv2.fillConvexPoly(locationmap, area, (255, 255, 255))
        # cv2.imshow('loc', locationmap)
        # cv2.waitKey(0)
        # locationmap = self.preprocess(int(self.resize_w / 16), int(self.resize_h / 16), locationmap, self.scale, 1)
        # locationmap = cv2.GaussianBlur(locationmap[0, :, :], (45, 45), 45)  # kernel_size=(45, 45)， sigma=45
        #
        # locationmap = np.expand_dims(locationmap, axis=0)
        # locationmap = np.array(locationmap)
        locationmap = LocationMapResize(h, w, points_all, self.resize_h, self.resize_w, self.coco)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, self.scale, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'areamap': torch.from_numpy(locationmap).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  # ,
        }


class DatasetPose_WDAGAN(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, dir_label, dir_center, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.csv_path = dir_label
        self.dir_center = dir_center
        self.scale = scale
        self.num_points = num_points
        # 读true_img标签
        with open(self.csv_path, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            print('len', len(self.labels))
            self.name_list = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels) - 1

            while img_sort < trainset_num:
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} true examples')

        # 读fake_img标签
        with open(self.dir_center, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.labels_fake = list(reader_fake)
            print('len', len(self.labels_fake))
            self.fakename_list = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            fake_sort = 1
            fake_num = len(self.labels_fake) - 9

            while fake_sort < fake_num:
                self.fakename_list.append(fake_sort)
                fake_sort = fake_sort + 1
            logging.info(f'Creating dataset with {len(self.fakename_list)} fake examples')
    def __len__(self):
        return max(len(self.labels) - 2, len(self.labels_fake) - 10)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):
        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        # if i == 0:
        #     print('i:', i, self.labels[i])
        # if i > 4000:
        #     print('i:', i)
        # 读取real images
        if i >= len(self.labels) - 2:
            ii = random.sample(range(len(self.labels) - 2), 1)[0]
        else:
            ii = i + 1
        Row = self.labels[ii]
        img_name = self.imgs_dir + Row[0]
        ####加数据增广版本
        img = Image.open(img_name)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)

        # 读取fake imgs
        if i >= len(self.labels_fake) - 10:
            ii = random.sample(range(len(self.labels_fake) - 10), 1)[0]
        else:
            ii = i
        idx = int(self.fakename_list[ii])
        # people_num = int(self.labels_fake[idx][1])  # 当前图像中的人数
        box = self.labels_fake[idx][2]  # 读取标记框坐标
        # print('box:', box)

        img_file = self.imgs_dir + self.labels_fake[idx][0]  # 获取图像名，读图
        fake = Image.open(img_file)
        fake0 = cv2.cvtColor(np.asarray(fake), cv2.COLOR_RGB2BGR)
        h, w, _ = fake0.shape
        fake = self.preprocess(self.resize_w, self.resize_h, fake0, 1)

        # 根据box标签生成位置热力图
        AreaMap0 = box2patch(fake0, box, int(self.resize_w / self.scale), int(self.resize_h / self.scale))
        # AreaMap = cv2.resize(AreaMap0, (int(self.resize_w/(4 * self.scale)), int(self.resize_h/(4 * self.scale))))
        # print(AreaMap0.shape)
        AreaMap = np.array(AreaMap0 / 255)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'fake': torch.from_numpy(fake).type(torch.FloatTensor),
            'centermap': torch.from_numpy(AreaMap).type(torch.FloatTensor)}


class DatasetPose_UDAGAN(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, dir_label, dir_fake, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.csv_path = dir_label
        self.dir_fake = dir_fake
        self.scale = scale
        self.num_points = num_points
        # 读true_img标签
        with open(self.csv_path, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            print('len', len(self.labels))
            self.name_list = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels) - 1

            while img_sort < trainset_num:
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} true examples')

        # 读fake_img标签
        with open(self.dir_fake, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.labels_fake = list(reader_fake)
            print('len', len(self.labels_fake))
            self.fakename_list = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            fake_sort = 1
            fake_num = len(self.labels_fake) - 9

            while fake_sort < fake_num:
                self.fakename_list.append(fake_sort)
                fake_sort = fake_sort + 1
            logging.info(f'Creating dataset with {len(self.fakename_list)} fake examples')
    def __len__(self):
        return max(len(self.labels) - 2, len(self.labels_fake) - 10)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):
        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        # 读取fake images
        if i >= len(self.labels_fake) - 10:
            ii = random.sample(range(1, len(self.labels_fake) - 10), 1)[0]
        else:
            ii = i + 1
        Row = self.labels_fake[ii]
        img_name = self.imgs_dir + Row[0]
        ####加数据增广版本
        fake = Image.open(img_name)
        fake0 = cv2.cvtColor(np.asarray(fake), cv2.COLOR_RGB2BGR)
        fake = self.preprocess(self.resize_w, self.resize_h, fake0, 1)

        # 读取real imgs
        if i >= len(self.labels) - 2:
            ii = random.sample(range(len(self.labels) - 2), 1)[0]
        else:
            ii = i
        idx = int(self.name_list[ii])
        box = self.labels[idx][2]
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        img = Image.open(img_file)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)

        # 根据box标签生成位置热力图
        AreaMap0 = box2patch(img0, box, int(self.resize_w / self.scale), int(self.resize_h / self.scale))
        # AreaMap = cv2.resize(AreaMap0, (int(self.resize_w/(4 * self.scale)), int(self.resize_h/(4 * self.scale))))
        # print(AreaMap0.shape)
        AreaMap = np.array(AreaMap0 / 255)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'fake': torch.from_numpy(fake).type(torch.FloatTensor),
            'centermap': torch.from_numpy(AreaMap).type(torch.FloatTensor)}


class DatasetPose_GAN(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, dir_label, dir_fake):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.csv_path = dir_label
        self.dir_fake = dir_fake

        # 读true_img标签
        with open(self.csv_path, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.name_list = list(reader)
            print('len', len(self.name_list))
            self.labels = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.name_list)

            while img_sort < trainset_num:
                self.labels.append(self.name_list[img_sort])
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.labels)} true examples')

        # 读fake_img文件名
        with open(self.dir_fake, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.name_list_f = list(reader)
            print('len', len(self.name_list_f))
            self.labels_f = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.name_list_f)

            while img_sort < trainset_num:
                self.labels_f.append(self.name_list_f[img_sort])
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.labels_f)} true examples')
    def __len__(self):
        return max(len(self.labels), len(self.labels_f))

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img):
        pil_img = cv2.resize(pil_img, (resize_w, resize_h))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd0 = img_nd
            img_nd = np.expand_dims(img_nd0, axis=2)
            img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
        img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        # 读取real images
        if i >= len(self.labels):
            ii = random.sample(range(len(self.labels)), 1)[0]
        else:
            ii = i
        Row = self.labels[ii]
        img_name = self.imgs_dir + Row[0]
        ####加数据增广版本
        img = Image.open(img_name)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img)

        # 读取fake imgs
        if i >= len(self.labels_f):
            ii = random.sample(range(len(self.labels_f)), 1)[0]
        else:
            ii = i
        Row_f = self.labels_f[ii]
        fake_name = self.imgs_dir + Row_f[0]
        # print('img_name:', img_name, 'fake_name:', fake_name)
        ####加数据增广版本
        fake = Image.open(fake_name)
        fake0 = cv2.cvtColor(np.asarray(fake), cv2.COLOR_RGB2BGR)
        fake = AugImg(fake0, [self.resize_h, self.resize_w])
        fake = self.preprocess(self.resize_w, self.resize_h, fake)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'fake': torch.from_numpy(fake).type(torch.FloatTensor)}


class DatasetPose_WDA_onlyCenter2(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, dir_center, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.dir_center = dir_center
        self.scale = scale
        self.num_points = num_points

        # 读fake_img标签
        with open(self.dir_center, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.labels_fake = list(reader_fake)
            print('len', len(self.labels_fake))
            self.fakename_list = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            fake_sort = 1
            fake_num = len(self.labels_fake) - 9

            while fake_sort < fake_num:
                self.fakename_list.append(fake_sort)
                fake_sort = fake_sort + 1
            logging.info(f'Creating dataset with {len(self.fakename_list)} fake examples')
    def __len__(self):
        return len(self.labels_fake) - 10

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):
        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        # 读取fake imgs
        if i >= len(self.labels_fake) - 10:
            ii = random.sample(range(len(self.labels_fake) - 10), 1)[0]
        else:
            ii = i
        idx = int(self.fakename_list[ii])
        box = self.labels_fake[idx][2]
        people_num = int(self.labels_fake[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels_fake[idx][0]  # 获取图像名，读图
        fake = Image.open(img_file)
        fake0 = cv2.cvtColor(np.asarray(fake), cv2.COLOR_RGB2BGR)
        h, w, _ = fake0.shape
        fake = self.preprocess(self.resize_w, self.resize_h, fake0, 1)

        # 根据box标签生成位置热力图
        AreaMap0 = box2patch(fake0, box, int(self.resize_w / self.scale), int(self.resize_h / self.scale))
        # AreaMap = cv2.resize(AreaMap0, (int(self.resize_w/(4 * self.scale)), int(self.resize_h/(4 * self.scale))))
        # print(AreaMap0.shape)
        AreaMap = np.array(AreaMap0 / 255)

        # shows = np.zeros([self.resize_h, self.resize_w, 3])
        # print('shape:', fake.shape, centermap0.shape)
        # for kk in range(3):
        #     shows[:, :, kk] = centermap0 + fake[kk, :, :] * 0.3
        # cv2.imshow('image', shows)
        # cv2.waitKey(0)
        #
        # print('shape:', centermap.shape, img.shape)

        return {
            'fake': torch.from_numpy(fake).type(torch.FloatTensor),
            'centermap': torch.from_numpy(AreaMap).type(torch.FloatTensor)}


class DatasetPoseCSV_pad(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points, aug):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points
        self.Aug = aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        # print(self.labels[idx][:9])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        img = Image.open(img_file)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape

        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img, 1)

        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point

                sigma = 3
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  #,
        }


class DatasetPoseCSV_pad_180(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points, aug):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points
        self.Aug = aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                self.name_list.append(img_sort)
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        # print(self.labels[idx][:9])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        img = Image.open(img_file)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        if i % 2 == 0:
            img0 = rotate_img(img0, 180, [w/2, h/2])
        # img = AugImg(img0, [self.resize_h, self.resize_w])

        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)

        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                # if i % 2 == 0:
                #     point = rotate_coordinates(point, 180, [w/2, h/2])
                # point = Trans_point(point, h, w, [self.resize_h, self.resize_w])

                points_all[k, n, :] = point

                sigma = 3
                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
                if i % 2 == 0:
                    heatmap0 = rotate_img(heatmap0, 180, [self.resize_w / 2, self.resize_h / 2])
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  #,
        }


class DatasetPoseCSV_padAP10K(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points, aug):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points
        self.Aug = aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                # print(int(self.labels[img_sort][1]), self.labels[img_sort])
                if int(self.labels[img_sort][1]) <= 6:
                    self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        # print(self.labels[idx][:9])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        img = Image.open(img_file)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape

        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img, 1)

        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point

                sigma = 3
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  #,
        }


class DatasetPoseCSV_AP10K(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            Point_name = self.labels[0][3:]
            # print('len', len(self.labels))
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                if int(self.labels[img_sort][1]) <= 6:
                    self.name_list.append(img_sort)
                # points_num = int(self.labels[img_sort][1])  # 当前图像中的人数
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        # people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        # print(img_file)
        # img0 = cv2.imread(img_file)
        img = Image.open(img_file)
        RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
        if RandomRate[0] == 1:
            img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
            img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)
        # index = self.labels[idx][0].index('.')
        searchContext = "_"
        # boxList = [m.start() for m in re.finditer(searchContext, self.labels[idx][2])]  # 获取第k个人Bounding box标签
        # box = [float(self.labels[idx][2][0:boxList[0]]), float(self.labels[idx][2][boxList[0] + 1:boxList[1]]),
        #        float(self.labels[idx][2][boxList[1] + 1:boxList[2]]), float(self.labels[idx][2][boxList[2] + 1:])]
        # w_box, h_box = int(box[2]), int(box[3])
        # 获取图像中第k个人的17个Key points
        # heatmaps = np.zeros([h, w, self.num_points7])
        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([1, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        index = idx
        for n in range(self.num_points):
            numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
            point = [int(self.labels[index][n + 3][0:numList[0]]),
                     int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
            points_all[0, n, :] = point
            # print(point)
            # sigma = max(int(h_box / 100), 1)  # 根据图中每个人的box高度确定高斯核参数sigma
            # sigma = max(int(w_box / 50), 1)
            sigma = 3

            heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
            heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
            heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

            # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
            # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
        shows = np.zeros([self.resize_h, self.resize_w, 3])
        print(img.shape)
        for kk in range(3):
            shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
        cv2.imshow('image', shows)
        cv2.waitKey(0)


        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps = np.array(heatmaps)
        # print('img:', img.shape)
        # print('heatmaps:', heatmaps.shape)
        # print(img.shape)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  #,
        }


class DatasetStage2_single(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, num_points, aug, scale, angle):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.angle = angle
        self.num_points = num_points
        self.Aug = aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_name = self.labels[idx][0]
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        numList = [m.start() for m in re.finditer('/', img_name)]
        if img_file[-9:-5] == 'flip':
            if img_name[1: numList[1]] in (['maze', 'minefield', 'treadmill']):
                img = Image.open(img_file[:-10] + '.bmp')
            else:
                img = Image.open(img_file[:-10] + '.jpg')
            if self.Aug == 1:
                RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
                if RandomRate[0] == 1:
                    img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                    img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape
            # img0 = cv2.flip(img0, int(img_file[-5]))
            img0 = rotate_img(img0, self.angle, [w / 2, h / 2])  # 旋转增广

        else:
            img = Image.open(img_file)
            if self.Aug == 1:
                RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
                if RandomRate[0] == 1:
                    img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                    img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape

        img = AugImg(img0, [self.resize_h, self.resize_w])
        # flip_type = 1 - random.sample(range(3), 1)[0]  # 随机生成-1到1之间的三个整数
        # img_flip = cv2.flip(img, flip_type)
        # print('angle:', self.angle)

        img = self.preprocess(self.resize_w, self.resize_h, img, 1)
        searchContext = "_"

        # 初始化heatmap
        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point

                sigma = 3
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # heatmap_flip = cv2.flip(heatmap, flip_type)
                # heatmaps_flip[:, :, n] = heatmaps_flip[:, :, n] + heatmap_flip

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor),
        }


class DatasetStage2_pad(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, num_points, aug, scale, angle):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.angle = angle
        self.num_points = num_points
        self.Aug = aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        # print(img_file, img_file[:-10], int(img_file[-5]))
        if img_file[-9:-5] == 'flip':
            img = Image.open(img_file[:-10] + '.jpg')
            if self.Aug == 1:
                RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
                if RandomRate[0] == 1:
                    img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                    img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape
            # img0 = cv2.flip(img0, int(img_file[-5]))
            img0 = rotate_img(img0, self.angle, [w / 2, h / 2])  # 旋转增广

        else:
            img = Image.open(img_file)
            if self.Aug == 1:
                RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
                if RandomRate[0] == 1:
                    img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                    img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape

        img = AugImg(img0, [self.resize_h, self.resize_w])
        # flip_type = 1 - random.sample(range(3), 1)[0]  # 随机生成-1到1之间的三个整数
        # img_flip = cv2.flip(img, flip_type)
        print('angle:', self.angle)
        img_rot0 = rotate_img(img0, self.angle, [w/2, h/2])
        img_rot = AugImg(img_rot0, [self.resize_h, self.resize_w])

        img = self.preprocess(self.resize_w, self.resize_h, img, 1)
        img_rot = self.preprocess(self.resize_w, self.resize_h, img_rot, 1)

        searchContext = "_"

        # 初始化heatmap
        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        heatmaps_rot = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]) * w / 640,
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]]) * h / 480]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point

                sigma = 3
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # heatmap_flip = cv2.flip(heatmap, flip_type)
                # heatmaps_flip[:, :, n] = heatmaps_flip[:, :, n] + heatmap_flip
                heatmap_rot = rotate_img(heatmap, self.angle,
                                         [int(self.resize_w / (2 * self.scale)), int(self.resize_h / (2 * self.scale))])
                heatmaps_rot[:, :, n] = heatmaps_rot[:, :, n] + heatmap_rot

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            shows = np.zeros([self.resize_h, self.resize_w, 3])
            shows_f = np.zeros([self.resize_h, self.resize_w, 3])
            heatmap0_rot = cv2.resize(heatmap_rot, (self.resize_w, self.resize_h))
            for kk in range(3):
                shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
                shows_f[:, :, kk] = heatmap0_rot + img_rot[kk, :, :] * 0.3
            cv2.imshow('image', shows)
            cv2.imshow('image_f', shows_f)
            cv2.waitKey(0)


        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)
        heatmaps_rot = self.preprocess(self.resize_w, self.resize_h, heatmaps_rot, 0)
        heatmaps_rot = np.array(heatmaps_rot)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'image_f': torch.from_numpy(img_rot).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor),
            'heatmap_f': torch.from_numpy(heatmaps_rot).type(torch.FloatTensor)
        }


class DatasetStage2_contrast(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, num_points, scale):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        # 读true_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.labels = self.labels[1:]
            print('len', len(self.labels))
            trainset_num = len(self.labels)
            # self.name_list = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            # img_sort = 1
            # trainset_num = len(self.labels)
            #
            # while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
            #     self.name_list.append(img_sort)
            #     img_sort = img_sort + 1
            logging.info(f'Creating dataset with {trainset_num} examples')

        # 读fake_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.target_label = list(reader_fake)
            self.target_label = self.target_label[1:]
            print('len', len(self.target_label))
            fake_num = len(self.target_label)
            # self.name_list_t = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            # img_sort = 1
            #
            # while img_sort < fake_num:  # 4198:  # len(self.labels):  #
            #     self.name_list_t.append(img_sort)
            #     img_sort = img_sort + 1
            logging.info(f'Creating dataset with {fake_num} examples')

    def __len__(self):
        return max(len(self.labels) - 2, len(self.target_label) - 2)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):
        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        # 读取fake images
        if i >= len(self.target_label) - 2:
            ii = random.sample(range(1, len(self.target_label) - 2), 1)[0]
        else:
            ii = i + 1
        idx = ii
        # idx = int(self.name_list_t[ii])
        img_name = self.target_label[idx][0]
        numList = [m.start() for m in re.finditer('/', img_name)]
        obj_num_t = int(self.target_label[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.target_label[idx][0]  # 获取图像名，读图
        if img_file[-9:-5] == 'flip':
            if img_name[1: numList[1]] in (['maze', 'minefield', 'treadmill']):
                target = Image.open(img_file[:-10] + '.bmp')
            else:
                target = Image.open(img_file[:-10] + '.jpg')
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                target1 = transforms.ColorJitter(brightness=0.5)(target)  # 随机从0~2之间的亮度变化
                target = transforms.ColorJitter(contrast=1)(target1)  # 随机从0~2之间的对比度
            target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
            # target0 = cv2.flip(target0, int(img_file[-5]))
            h, w, _ = target0.shape
            target0 = rotate_img(target0, 180, [w / 2, h / 2])
        else:
            target = Image.open(img_file)
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                target1 = transforms.ColorJitter(brightness=0.5)(target)  # 随机从0~2之间的亮度变化
                target = transforms.ColorJitter(contrast=1)(target1)  # 随机从0~2之间的对比度
            target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
            h, w, _ = target0.shape
        # target0 = AugImg(target0, [self.resize_h, self.resize_w])
        target = self.preprocess(self.resize_w, self.resize_h, target0, 1)
        searchContext = "_"
        # 初始化heatmap
        heatmaps_t = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        points_all_t = np.zeros([obj_num_t, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(obj_num_t):
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.target_label[index][n + 3])]
                point_t = [int(self.target_label[index][n + 3][0:numList[0]]),
                           int(self.target_label[index][n + 3][numList[0] + 1:numList[1]])]
                # point_t = [int(self.target_label[index][n + 3][0:numList[0]]) * w / self.resize_w,
                #            int(self.target_label[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                # point_t = Trans_point(point_t, h, w, [self.resize_h, self.resize_w])
                points_all_t[k, n, :] = point_t
                # heatmap0_t = CenterLabelHeatMap(self.resize_h, self.resize_w, point_t[0], point_t[1], sigma=3)
                heatmap0_t = CenterLabelHeatMapResize(h, w, point_t[0], point_t[1], self.resize_h, self.resize_w, sigma=3)
                heatmap_t = cv2.resize(heatmap0_t, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))

                heatmaps_t[:, :, n] = heatmaps_t[:, :, n] + heatmap_t

        # 读取real imgs 并生成heatmap
        if i >= len(self.labels) - 2:
            ii = random.sample(range(len(self.labels) - 2), 1)[0]
        else:
            ii = i
        idx = ii
        # idx = int(self.name_list[ii])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        img = Image.open(img_file)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img, 1)

        # 初始化heatmap
        heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                # point = [int(self.labels[index][n + 3][0:numList[0]]) * w / self.resize_w,
                #            int(self.labels[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma=3)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # shows_t = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            #     shows_t[:, :, kk] = heatmap0_t + target[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.imshow('image_t', shows_t)
            # cv2.waitKey(0)

        heatmaps_t = self.preprocess(self.resize_w, self.resize_h, heatmaps_t, 0)
        heatmaps_t = np.array(heatmaps_t)
        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        RandomRate2 = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
        if RandomRate2[0] == 1:
            return {
                'image_s': torch.from_numpy(img).type(torch.FloatTensor),
                'image_t': torch.from_numpy(target).type(torch.FloatTensor),
                'heatmap_s': torch.from_numpy(heatmaps).type(torch.FloatTensor),
                'heatmap_t': torch.from_numpy(heatmaps_t).type(torch.FloatTensor)
            }
        else:
            return {
                'image_s': torch.from_numpy(target).type(torch.FloatTensor),
                'image_t': torch.from_numpy(img).type(torch.FloatTensor),
                'heatmap_s': torch.from_numpy(heatmaps_t).type(torch.FloatTensor),
                'heatmap_t': torch.from_numpy(heatmaps).type(torch.FloatTensor)
            }


class DatasetStage2_update(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, num_points, scale, angle):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.angle = angle
        self.num_points = num_points
        # 读true_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.labels = self.labels[1:]
            print('len', len(self.labels))
            trainset_num = len(self.labels)
            logging.info(f'Creating dataset with {trainset_num} source examples')

        # 读target_label标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.target_label = list(reader_fake)
            self.target_label = self.target_label[1:]
            print('len', len(self.target_label))
            Target_num = len(self.target_label)
            logging.info(f'Creating dataset with {Target_num} target examples')

    def __len__(self):
        return max(len(self.labels) - 2, len(self.target_label) - 2)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):
        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        searchContext = "_"
        # 读取target images
        if i >= len(self.target_label) - 2:
            idx = random.sample(range(1, len(self.target_label) - 2), 1)[0]
        else:
            idx = i + 1
        box = self.target_label[idx][2]
        # 根据box的最大值判断是否存在伪标签
        numList = [m.start() for m in re.finditer(searchContext, box)]
        box = [int(box[0:numList[0]]), int(box[numList[0] + 1:numList[1]]),
               int(box[numList[1] + 1:numList[2]]), int(box[numList[2] + 1:])]
        if max(box) == 0:  # 无伪标签时，读取图像并进行随机增广，然后进行自监督训练
            SelfSupervision = np.ones(1)
            img_file = self.imgs_dir + self.target_label[idx][0]  # 获取图像名，读图
            img0 = Image.open(img_file)
            img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
            h, w, _ = img.shape
            img = AugImg(img, [self.resize_h, self.resize_w])
            img = self.preprocess(self.resize_w, self.resize_h, img, 1)

            # 通过增广构建另一个输入图像
            target = transforms.ColorJitter(brightness=0.5)(img0)  # 随机从0~2之间的亮度变化
            target = transforms.ColorJitter(contrast=1)(target)  # 随机从0~2之间的对比度
            target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)

            target0 = rotate_img(target0, self.angle, [w / 2, h / 2])  # 旋转增广
            target = AugImg(target0, [self.resize_h, self.resize_w])
            target = self.preprocess(self.resize_w, self.resize_h, target, 1)

            heatmaps = np.zeros([self.num_points, int(self.resize_h / self.scale), int(self.resize_w / self.scale)])
            heatmaps_t = np.zeros([self.num_points, int(self.resize_h / self.scale), int(self.resize_w / self.scale)])

        else:
            SelfSupervision = np.zeros(1)
            img_file = self.imgs_dir + self.target_label[idx][0]  # 获取图像名，读图
            if img_file[-9:-5] == 'flip':
                target = Image.open(img_file[:-10] + '.jpg')
                RandomRate = random.sample(range(3), 1)  # 百分之三十的概率进行数据增广
                if RandomRate[0] == 1:
                    target1 = transforms.ColorJitter(brightness=0.5)(target)  # 随机从0~2之间的亮度变化
                    target = transforms.ColorJitter(contrast=1)(target1)  # 随机从0~2之间的对比度
                target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
                target0 = cv2.flip(target0, int(img_file[-5]))
            else:
                target = Image.open(img_file)
                RandomRate = random.sample(range(3), 1)  # 百分之三十的概率进行数据增广
                if RandomRate[0] == 1:
                    target1 = transforms.ColorJitter(brightness=0.5)(target)  # 随机从0~2之间的亮度变化
                    target = transforms.ColorJitter(contrast=1)(target1)  # 随机从0~2之间的对比度
                target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
            h, w, _ = target0.shape
            target = AugImg(target0, [self.resize_h, self.resize_w])
            target = self.preprocess(self.resize_w, self.resize_h, target, 1)

            # 初始化heatmap
            heatmaps_t = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
            points_all_t = np.zeros([1, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.target_label[index][n + 3])]
                # point_t = [int(self.target_label[index][n + 3][0:numList[0]]) * w / self.resize_w,
                #            int(self.target_label[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                point_t = [int(self.target_label[index][n + 3][0:numList[0]]) * w / self.resize_w,
                           int(self.target_label[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                point_t = Trans_point(point_t, h, w, [self.resize_h, self.resize_w])
                points_all_t[0, n, :] = point_t
                heatmap0_t = CenterLabelHeatMap(self.resize_h, self.resize_w, point_t[0], point_t[1], sigma=3)
                heatmap_t = cv2.resize(heatmap0_t, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps_t[:, :, n] = heatmaps_t[:, :, n] + heatmap_t

            # 读取real imgs 并生成heatmap
            if i >= len(self.labels) - 2:
                idx = random.sample(range(len(self.labels) - 2), 1)[0]
            else:
                idx = i
            img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
            img = Image.open(img_file)
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape
            img = AugImg(img0, [self.resize_h, self.resize_w])
            img = self.preprocess(self.resize_w, self.resize_h, img, 1)

            # 初始化heatmap
            heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
            points_all = np.zeros([1, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                # point = [int(self.labels[index][n + 3][0:numList[0]]) * w / self.resize_w,
                #            int(self.labels[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[0, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma=3)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # shows_t = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            #     shows_t[:, :, kk] = heatmap0_t + target[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.imshow('image_t', shows_t)
            # cv2.waitKey(0)
            heatmaps_t = self.preprocess(self.resize_w, self.resize_h, heatmaps_t, 0)
            heatmaps_t = np.array(heatmaps_t)
            heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
            heatmaps = np.array(heatmaps)

        return {
            'image_s': torch.from_numpy(img).type(torch.FloatTensor),
            'image_t': torch.from_numpy(target).type(torch.FloatTensor),
            'heatmap_s': torch.from_numpy(heatmaps).type(torch.FloatTensor),
            'heatmap_t': torch.from_numpy(heatmaps_t).type(torch.FloatTensor),
            'self_supvision': torch.from_numpy(SelfSupervision).type(torch.FloatTensor)
        }


class DatasetStage2_iteration(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, target_unlabel, num_points, scale, angle):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.target_unlabel = target_unlabel
        self.scale = scale
        self.angle = angle
        self.num_points = num_points
        # 读true_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.labels = self.labels[1:]
            print('len', len(self.labels))
            trainset_num = len(self.labels)
            logging.info(f'Creating dataset with {trainset_num} source examples')

        # 读target_label标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.target_label = list(reader_fake)
            self.target_label = self.target_label[1:]
            print('len_target_label', len(self.target_label))
            Target_num = len(self.target_label)
            logging.info(f'Creating dataset with {Target_num} target examples with peusdo label')

        # 读target_unlabel数据
        with open(self.target_unlabel, 'r', encoding='gb18030') as fff:
            reader_unlabel = csv.reader(fff)
            self.target_unlabel = list(reader_unlabel)[1:]

            random.shuffle(self.target_unlabel)
            Unlabel_num = len(self.target_unlabel)
            print('len_target_unlabel', Unlabel_num)

            logging.info(f'Creating dataset with {Unlabel_num} unlabeled target examples')
            if Unlabel_num > Target_num / 4:
                self.target_unlabel = self.target_unlabel[:int(Target_num / 4)]
                Unlabel_num = int(Target_num / 4)
            for l in range(Unlabel_num):
                self.target_label.append(self.target_unlabel[l])
            logging.info(f'Creating dataset with {len(self.target_label)} target examples in total')

    def __len__(self):
        # return 1000
        return max(len(self.labels) - 2, len(self.target_label) - 2)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):
        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        searchContext = "_"
        # 读取target images
        if i >= len(self.target_label) - 2:
            idx = random.sample(range(1, len(self.target_label) - 2), 1)[0]
        else:
            idx = i + 1
        box = self.target_label[idx][2]
        # 根据box的最大值判断是否存在伪标签
        numList = [m.start() for m in re.finditer(searchContext, box)]
        box = [int(box[0:numList[0]]), int(box[numList[0] + 1:numList[1]]),
               int(box[numList[1] + 1:numList[2]]), int(box[numList[2] + 1:])]
        if max(box) == 0:  # 无伪标签时，读取图像并进行随机增广，然后进行自监督训练
            SelfSupervision = np.ones(1)
            img_file = self.imgs_dir + self.target_label[idx][0]  # 获取图像名，读图
            img0 = Image.open(img_file)
            img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
            h, w, _ = img.shape
            img = AugImg(img, [self.resize_h, self.resize_w])
            img = self.preprocess(self.resize_w, self.resize_h, img, 1)

            # 通过增广构建另一个输入图像
            target = transforms.ColorJitter(brightness=0.5)(img0)  # 随机从0~2之间的亮度变化
            target = transforms.ColorJitter(contrast=1)(target)  # 随机从0~2之间的对比度
            target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)

            target0 = rotate_img(target0, self.angle, [w / 2, h / 2])  # 旋转增广
            target = AugImg(target0, [self.resize_h, self.resize_w])
            target = self.preprocess(self.resize_w, self.resize_h, target, 1)

            heatmaps = np.zeros([self.num_points, int(self.resize_h / self.scale), int(self.resize_w / self.scale)])
            heatmaps_t = np.zeros([self.num_points, int(self.resize_h / self.scale), int(self.resize_w / self.scale)])

        else:
            SelfSupervision = np.zeros(1)
            img_name = self.target_label[idx][0]
            numList = [m.start() for m in re.finditer('/', img_name)]
            img_file = self.imgs_dir + self.target_label[idx][0]  # 获取图像名，读图
            # print(self.target_label[idx][0])
            if img_file[-9:-5] == 'flip':
                if img_name[1: numList[1]] in (['maze', 'minefield', 'treadmill']):
                    target = Image.open(img_file[:-10] + '.bmp')
                else:
                    target = Image.open(img_file[:-10] + '.jpg')
                RandomRate = random.sample(range(3), 1)  # 百分之三十的概率进行数据增广
                if RandomRate[0] == 1:
                    target1 = transforms.ColorJitter(brightness=0.5)(target)  # 随机从0~2之间的亮度变化
                    target = transforms.ColorJitter(contrast=1)(target1)  # 随机从0~2之间的对比度
                target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
                h, w, _ = target0.shape
                # target0 = cv2.flip(target0, int(img_file[-5]))
                # target0 = rotate_img(target0, self.angle, [w / 2, h / 2])  # 旋转增广
                target0 = cv2.flip(cv2.flip(target0, 1), 0)

            else:
                target = Image.open(img_file)
                RandomRate = random.sample(range(3), 1)  # 百分之三十的概率进行数据增广
                if RandomRate[0] == 1:
                    target1 = transforms.ColorJitter(brightness=0.5)(target)  # 随机从0~2之间的亮度变化
                    target = transforms.ColorJitter(contrast=1)(target1)  # 随机从0~2之间的对比度
                target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
                h, w, _ = target0.shape
            target = AugImg(target0, [self.resize_h, self.resize_w])
            target = self.preprocess(self.resize_w, self.resize_h, target, 1)

            # 初始化heatmap
            heatmaps_t = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
            points_all_t = np.zeros([1, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.target_label[index][n + 3])]
                point_t = [int(self.target_label[index][n + 3][0:numList[0]]),
                           int(self.target_label[index][n + 3][numList[0] + 1:numList[1]])]
                # point_t = [int(self.target_label[index][n + 3][0:numList[0]]) * w / self.resize_w,
                #            int(self.target_label[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                point_t = Trans_point(point_t, h, w, [self.resize_h, self.resize_w])
                points_all_t[0, n, :] = point_t
                heatmap0_t = CenterLabelHeatMap(self.resize_h, self.resize_w, point_t[0], point_t[1], sigma=3)
                heatmap_t = cv2.resize(heatmap0_t, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps_t[:, :, n] = heatmaps_t[:, :, n] + heatmap_t

            # 读取real imgs 并生成heatmap
            if i >= len(self.labels) - 2:
                idx = random.sample(range(len(self.labels) - 2), 1)[0]
            else:
                idx = i
            img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
            img = Image.open(img_file)
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape
            img = AugImg(img0, [self.resize_h, self.resize_w])
            img = self.preprocess(self.resize_w, self.resize_h, img, 1)

            # 初始化heatmap
            heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
            points_all = np.zeros([1, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                # point = [int(self.labels[index][n + 3][0:numList[0]]) * w / self.resize_w,
                #            int(self.labels[index][n + 3][numList[0] + 1:numList[1]]) * h / self.resize_h]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[0, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma=3)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # shows_t = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     # shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            #     shows_t[:, :, kk] = heatmap0_t + target[kk, :, :] * 0.3
            # # cv2.imshow('image', shows)
            # cv2.imshow('image_t', shows_t)
            # cv2.waitKey(0)
            heatmaps_t = self.preprocess(self.resize_w, self.resize_h, heatmaps_t, 0)
            heatmaps_t = np.array(heatmaps_t)
            heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
            heatmaps = np.array(heatmaps)

        return {
            'image_s': torch.from_numpy(img).type(torch.FloatTensor),
            'image_t': torch.from_numpy(target).type(torch.FloatTensor),
            'heatmap_s': torch.from_numpy(heatmaps).type(torch.FloatTensor),
            'heatmap_t': torch.from_numpy(heatmaps_t).type(torch.FloatTensor),
            'self_supvision': torch.from_numpy(SelfSupervision).type(torch.FloatTensor)
        }
