# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: Contrastive learning
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2023-02-16
# --------------------------------------------------------
"""

import re
import csv
import cv2
import torch
import random
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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


def box2map(Img):  # https://blog.csdn.net/qq_30154571/article/details/109557559
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, 1, 5)
    # blur = cv2.blur(dist_transform, (5, 5))
    # alpha = 255 / (blur.max())
    return dist_transform


def box2patch(Image, Annotation, resizeW, resizeH):
    H, W, C = Image.shape
    # Image = cv2.resize(Image, (resizeW, resizeH))
    temp = np.zeros([resizeH, resizeW])

    numList = [m.start() for m in re.finditer('_', Annotation)]
    lu_x0 = int(float(Annotation[0:numList[0]]) * (resizeW / W))
    lu_y0 = int(float(Annotation[numList[0] + 1:numList[1]]) * (resizeH / H))
    rb_x0 = int(float(Annotation[numList[1] + 1:numList[2]]) * (resizeW / W))
    rb_y0 = int(float(Annotation[numList[2] + 1:]) * (resizeH / H))
    # print('box:', Annotation)

    patch = Image[min(lu_y0, rb_y0):max(lu_y0, rb_y0), min(lu_x0, rb_x0):max(lu_x0, rb_x0), :]
    temp[min(lu_y0, rb_y0):max(lu_y0, rb_y0), min(lu_x0, rb_x0):max(lu_x0, rb_x0)] = box2map(patch)

    blur = cv2.blur(temp, (3, 3))
    alpha = 255 / (blur.max())
    blur = blur * alpha
    # cv2.imshow('img_new', Image)
    # cv2.imshow('img_new2', np.uint8(blur))
    # cv2.waitKey(0)
    blur = cv2.resize(blur, (int(resizeW / 4), int(resizeH / 4)))
    blur = np.expand_dims(blur, axis=0)
    return blur


def AugImg(Img, resize):
    h, w, _ = Img.shape
    if max([h, w]) == w:
        res_2 = int(h * (resize[1] / w))
        Img = cv2.resize(Img, (resize[1], res_2))
        padding_l = int((resize[0] - res_2) / 2)
        Img = cv2.copyMakeBorder(Img, padding_l, padding_l, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    else:
        res_2 = int(w * (resize[0] / h))
        Img = cv2.resize(Img, (res_2, resize[0]))
        padding_l = int((resize[1] - res_2) / 2)
        Img = cv2.copyMakeBorder(Img, 0, 0, padding_l, padding_l, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return Img


def Trans_point(p, h, w, resize):
    if max([h, w]) == w:
        res_2 = int(h * (resize[1] / w))
        padding_l = int((resize[0] - res_2) / 2)
        p[0] = int(p[0] * (resize[1] / w))  # x
        p[1] = int(p[1] * (resize[1] / w) + padding_l)  # y
    else:
        res_2 = int(w * (resize[0] / h))
        padding_l = int((resize[1] - res_2) / 2)
        p[0] = int(p[0] * (resize[0] / h) + padding_l)  # x
        p[1] = int(p[1] * (resize[0] / h))  # y

    return p


class Dataset_contrast(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

        # 读fake_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_target = csv.reader(ff)
            self.labels_target = list(reader_target)
            print('len', len(self.labels_target))
            self.namelist_target = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            target_sort = 1
            target_num = len(self.labels_target)

            while target_sort < target_num:
                self.namelist_target.append(target_sort)
                target_sort = target_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_target)} fake examples')
    def __len__(self):
        return max(len(self.labels_source) - 2, len(self.labels_target) - 2)

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
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        sourceimg_name = self.imgs_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img0, 1)
        # 根据标签生成关键点热力图
        heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_source[index][n + 3])]
                point = [int(self.labels_source[index][n + 3][0:numList[0]]),
                         int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point
                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma=3)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap
        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps = np.array(heatmaps)

        # 读取target imgs并生成位置热力图
        # if i >= len(self.labels_target) - 2:
        #     ii = random.sample(range(len(self.labels_target) - 2), 1)[0]
        # else:
        #     ii = i
        sort = random.sample(range(len(self.labels_target) - 2), 1)[0]
        idx = int(self.namelist_target[sort])
        box = self.labels_target[idx][2]
        targetimg_name = self.imgs_dir + self.labels_target[idx][0]  # 获取图像名，读图
        # print('img:', sourceimg_name, targetimg_name)
        target_img = Image.open(targetimg_name)
        target_img0 = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
        target_img = self.preprocess(self.resize_w, self.resize_h, target_img0, 1)
        # 根据box标签生成位置热力图
        AreaMap0 = box2patch(target_img0, box, int(self.resize_w / self.scale), int(self.resize_h / self.scale))
        AreaMap = np.array(AreaMap0 / 255)

        return {
            'source_img': torch.from_numpy(source_img).type(torch.FloatTensor),  # source-->heatmap
            'target_img': torch.from_numpy(target_img).type(torch.FloatTensor),  # target-->areamap;
            'areamap': torch.from_numpy(AreaMap).type(torch.FloatTensor),
            'heatmaps': torch.from_numpy(heatmaps).type(torch.FloatTensor)}


class Dataset_contrast_2hms(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

        # 读target_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_target = csv.reader(ff)
            self.labels_target = list(reader_target)
            print('len', len(self.labels_target))
            self.namelist_target = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            target_sort = 1
            target_num = len(self.labels_target)

            while target_sort < target_num:
                self.namelist_target.append(target_sort)
                target_sort = target_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_target)} fake examples')
    def __len__(self):
        return max(len(self.labels_source) - 2, len(self.labels_target) - 2)

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
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        sourceimg_name = self.imgs_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img0, 1)
        # 根据标签生成关键点热力图
        heatmaps_1 = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_source[index][n + 3])]
                point = [int(self.labels_source[index][n + 3][0:numList[0]]),
                         int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]])]
                heatmap0_1 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma=3)
                heatmap_1 = cv2.resize(heatmap0_1, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps_1[:, :, n] = heatmaps_1[:, :, n] + heatmap_1
        heatmaps_1 = self.preprocess(self.resize_w, self.resize_h, heatmaps_1, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps_1 = np.array(heatmaps_1)

        show_source = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_source[:, :, kk] = heatmap0_1 + source_img[kk, :, :] * 0.3

        # 读取target imgs并生成位置热力图
        # if i >= len(self.labels_target) - 2:
        #     ii = random.sample(range(len(self.labels_target) - 2), 1)[0]
        # else:
        #     ii = i
        sort = random.sample(range(len(self.labels_target) - 2), 1)[0]
        idx = int(self.namelist_target[sort])
        people_num_t = int(self.labels_target[idx][1])  # 当前图像中的人数
        targetimg_name = self.imgs_dir + self.labels_target[idx][0]  # 获取图像名，读图
        # print('img:', sourceimg_name, targetimg_name)
        target_img = Image.open(targetimg_name)
        target_img0 = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
        h, w, _ = target_img0.shape
        target_img = self.preprocess(self.resize_w, self.resize_h, target_img0, 1)
        # 根据关键点标签生成中心点热力图
        heatmaps_2 = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        for k in range(people_num_t):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_target[index][n + 3])]
                point = [int(self.labels_target[index][n + 3][0:numList[0]]),
                         int(self.labels_target[index][n + 3][numList[0] + 1:numList[1]])]
                heatmap0_2 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma=3)
                heatmap_2 = cv2.resize(heatmap0_2, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps_2[:, :, n] = heatmaps_2[:, :, n] + heatmap_2
        heatmaps_2 = self.preprocess(self.resize_w, self.resize_h, heatmaps_2, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps_2 = np.array(heatmaps_2)

        show_target = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_target[:, :, kk] = heatmap0_2 + target_img[kk, :, :] * 0.3

        cv2.imshow('show_source', show_source)
        cv2.imshow('show_target', show_target)
        cv2.waitKey(0)

        return {
            'img_1': torch.from_numpy(source_img).type(torch.FloatTensor),  # source-->heatmap
            'img_2': torch.from_numpy(target_img).type(torch.FloatTensor),  # target-->areamap;
            'heatmaps_1': torch.from_numpy(heatmaps_1).type(torch.FloatTensor),
            'heatmaps_2': torch.from_numpy(heatmaps_2).type(torch.FloatTensor)}


class Dataset_contrast_center(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, scale, num_points, Aug):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        self.Aug = Aug
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

        # 读fake_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_target = csv.reader(ff)
            self.labels_target = list(reader_target)
            print('len', len(self.labels_target))
            self.namelist_target = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            target_sort = 1
            target_num = len(self.labels_target)

            while target_sort < target_num:
                self.namelist_target.append(target_sort)
                target_sort = target_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_target)} fake examples')
    def __len__(self):
        return max(len(self.labels_source) - 2, len(self.labels_target) - 2)

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
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        numList0 = [m.start() for m in re.finditer("/", self.labels_source[idx][0])]
        folder_img = self.labels_source[idx][0][numList0[0] + 1 : numList0[1]]
        if folder_img == 'Rat1Crops_v2':
            sigma = 3
        else:
            sigma = 2
        # sigma = 2
        sourceimg_name = self.imgs_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                source_img1 = transforms.ColorJitter(brightness=0.5)(source_img)  # 随机从0~2之间的亮度变化
                source_img = transforms.ColorJitter(contrast=1)(source_img1)  # 随机从0~2之间的对比度
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = AugImg(source_img0, [self.resize_h, self.resize_w])
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img, 1)
        # 根据标签生成关键点热力图
        heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_source[index][n + 3])]
                point = [int(self.labels_source[index][n + 3][0:numList[0]]),
                         int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap
        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps = np.array(heatmaps)

        show_source = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_source[:, :, kk] = heatmap0 + source_img[kk, :, :] * 0.3

        # 读取target imgs并生成位置热力图
        # if i >= len(self.labels_target) - 2:
        #     ii = random.sample(range(len(self.labels_target) - 2), 1)[0]
        # else:
        #     ii = i
        sort = random.sample(range(len(self.labels_target) - 2), 1)[0]
        idx = int(self.namelist_target[sort])
        # idx = i
        # print(self.labels_target[idx])
        people_num_t = int(self.labels_target[idx][1])  # 当前图像中的人数
        numList0 = [m.start() for m in re.finditer("/", self.labels_target[idx][0])]
        folder_img = self.labels_target[idx][0][numList0[0] + 1 : numList0[1]]
        # print(self.labels_target[idx][0], folder_img)
        if folder_img == 'VerticalCrops_v2':
            sigma = 8
        else:
            sigma = 3
        # sigma = 1

        targetimg_name = self.imgs_dir + self.labels_target[idx][0]  # 获取图像名，读图
        # print('img:', sourceimg_name, targetimg_name)
        target_img = Image.open(targetimg_name)
        target_img0 = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
        ht, wt, _ = target_img0.shape
        target_img = AugImg(target_img0, [self.resize_h, self.resize_w])
        target_img = self.preprocess(self.resize_w, self.resize_h, target_img, 1)
        # 根据关键点标签生成中心点热力图
        points_all_target = np.zeros([people_num_t, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num_t):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_target[index][n + 3])]
                point = [int(self.labels_target[index][n + 3][0:numList[0]]),
                         int(self.labels_target[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, ht, wt, [self.resize_h, self.resize_w])
                points_all_target[k, n, :] = point

        center = np.mean(points_all_target, axis=1)
        center = center[0]
        centermap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, center[0], center[1], sigma)
        centermap = cv2.resize(centermap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
        centermap = np.expand_dims(centermap, axis=0)
        centermap = np.array(centermap / 255)
        # print('maps:', heatmaps.shape, centermap.shape)

        show_target = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_target[:, :, kk] = centermap0 + target_img[kk, :, :] * 0.3
        cv2.imshow('show_source', show_source)
        cv2.imshow('show_target', show_target)
        cv2.imshow('hm', np.uint8(centermap0))
        cv2.waitKey(0)

        return {
            'image': torch.from_numpy(source_img).type(torch.FloatTensor),  # source-->heatmap
            'target_img': torch.from_numpy(target_img).type(torch.FloatTensor),  # target-->areamap;
            'centermap': torch.from_numpy(centermap).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)}


class Dataset_contrast_center_person(Dataset):
    def __init__(self, resize_w, resize_h, dir_img_s, source_label, dir_img_t, target_label, scale, num_points,
                 num_points_t):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.img_s_dir = dir_img_s
        self.img_t_dir = dir_img_t
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        self.num_points_t = num_points_t
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

        # 读fake_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_target = csv.reader(ff)
            self.labels_target = list(reader_target)
            print('len', len(self.labels_target))
            self.namelist_target = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            target_sort = 1
            target_num = len(self.labels_target)

            while target_sort < target_num:
                self.namelist_target.append(target_sort)
                target_sort = target_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_target)} fake examples')
    def __len__(self):
        return max(len(self.labels_source) - 2, len(self.labels_target) - 2)

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
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        sourceimg_name = self.img_s_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = AugImg(source_img0, [self.resize_h, self.resize_w])
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img0, 1)
        # 根据标签生成关键点热力图
        heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_source[index][n + 3])]
                point = [int(self.labels_source[index][n + 3][0:numList[0]]),
                         int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma=3)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap
        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps = np.array(heatmaps)
        show_source = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_source[:, :, kk] = heatmap0 + source_img[kk, :, :] * 0.3

        # 读取target imgs并生成位置热力图
        # if i >= len(self.labels_target) - 2:
        #     ii = random.sample(range(len(self.labels_target) - 2), 1)[0]
        # else:
        #     ii = i
        sort = random.sample(range(len(self.labels_target) - 2), 1)[0]
        idx = int(self.namelist_target[sort])
        people_num_t = int(self.labels_target[idx][1])  # 当前图像中的人数
        targetimg_name = self.img_t_dir + self.labels_target[idx][0]  # 获取图像名，读图
        # print('img:', sourceimg_name, targetimg_name)
        target_img = Image.open(targetimg_name)
        target_img0 = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
        ht, wt, _ = target_img0.shape
        target_img = AugImg(target_img0, [self.resize_h, self.resize_w])
        target_img = self.preprocess(self.resize_w, self.resize_h, target_img0, 1)
        # 根据关键点标签生成中心点热力图
        points_all_target = np.zeros([people_num_t, self.num_points_t, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num_t):
            index = idx + k
            for n in range(self.num_points_t):
                numList = [m.start() for m in re.finditer("_", self.labels_target[index][n + 3])]
                point = [int(self.labels_target[index][n + 3][0:numList[0]]),
                         int(self.labels_target[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, ht, wt, [self.resize_h, self.resize_w])
                points_all_target[k, n, :] = point

        center = np.mean(points_all_target, axis=1)
        center = center[0]
        sigma = 8
        centermap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, center[0], center[1], sigma)
        centermap = cv2.resize(centermap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
        centermap = np.expand_dims(centermap, axis=0)
        centermap = np.array(centermap / 255)

        show_target = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_target[:, :, kk] = centermap0 + target_img[kk, :, :] * 0.3
        cv2.imshow('show_source', show_source)
        cv2.imshow('show_target', show_target)
        cv2.imshow('hm', np.uint8(centermap0))
        cv2.waitKey(0)

        return {
            'image': torch.from_numpy(source_img).type(torch.FloatTensor),  # source-->heatmap
            'target_img': torch.from_numpy(target_img).type(torch.FloatTensor),  # target-->areamap;
            'centermap': torch.from_numpy(centermap).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)}


class Dataset_all_center(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

        # 读fake_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_target = csv.reader(ff)
            self.labels_target = list(reader_target)
            print('len', len(self.labels_target))
            self.namelist_target = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            target_sort = 1
            target_num = len(self.labels_target)

            while target_sort < target_num:
                self.namelist_target.append(target_sort)
                target_sort = target_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_target)} fake examples')
    def __len__(self):
        return max(len(self.labels_source) - 2, len(self.labels_target) - 2)

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
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        sourceimg_name = self.imgs_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img0, 1)

        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_source[index][n + 3])]
                point = [int(self.labels_source[index][n + 3][0:numList[0]]),
                         int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point

        center_source = np.mean(points_all, axis=1)
        center_source = center_source[0]
        sigma = 3
        centermap0_source = CenterLabelHeatMapResize(h, w, center_source[0], center_source[1],
                                                     self.resize_h, self.resize_w, sigma)
        centermap_source = cv2.resize(centermap0_source, (28, 28))
        centermap_source = np.expand_dims(centermap_source, axis=0)
        centermap_source = np.array(centermap_source / 255)

        # 读取target imgs并生成位置热力图
        if i >= len(self.labels_target) - 2:
            ii = random.sample(range(len(self.labels_target) - 2), 1)[0]
        else:
            ii = i
        # sort = random.sample(range(len(self.labels_target) - 2), 1)[0]
        # idx = int(self.namelist_target[sort])
        idx = int(self.namelist_target[ii])
        people_num_t = int(self.labels_target[idx][1])  # 当前图像中的人数
        targetimg_name = self.imgs_dir + self.labels_target[idx][0]  # 获取图像名，读图
        # print('img:', sourceimg_name, targetimg_name)
        target_img = Image.open(targetimg_name)
        target_img0 = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
        h, w, _ = target_img0.shape
        target_img = self.preprocess(self.resize_w, self.resize_h, target_img0, 1)
        # 根据关键点标签生成中心点热力图
        points_all_target = np.zeros([people_num_t, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for kt in range(people_num_t):
            index_t = idx + kt
            for nt in range(self.num_points):
                numList_t = [m.start() for m in re.finditer("_", self.labels_target[index_t][nt + 3])]
                point_target = [int(self.labels_target[index_t][nt + 3][0:numList_t[0]]),
                         int(self.labels_target[index_t][nt + 3][numList_t[0] + 1:numList_t[1]])]
                points_all_target[kt, nt, :] = point_target

        center_target = np.mean(points_all_target, axis=1)
        center_target = center_target[0]
        sigma = 3
        centermap0_target = CenterLabelHeatMapResize(h, w, center_target[0], center_target[1],
                                                     self.resize_h, self.resize_w, sigma)
        centermap_target = cv2.resize(centermap0_target, (28, 28))
        centermap_target = np.expand_dims(centermap_target, axis=0)
        centermap_target = np.array(centermap_target / 255)
        # print('maps:', heatmaps.shape, centermap.shape)
        
        # debug_可视化标签
        # print(source_img0.shape, centermap0_source.shape, source_img.shape)
        # source_show = (cv2.resize(source_img0, (self.resize_w, self.resize_h))) * 0.5
        # source_show[:, :, 1] = source_show[:, :, 1] + centermap0_source
        # target_show = (cv2.resize(target_img0, (self.resize_w, self.resize_h))) * 0.5
        # target_show[:, :, 1] = target_show[:, :, 1] + centermap0_target
        #
        # cv2.imshow('source:', np.uint8(source_show))
        # cv2.imshow('target:', np.uint8(target_show))
        # cv2.waitKey(0)

        return {
            'source_img': torch.from_numpy(source_img).type(torch.FloatTensor),  # source-->heatmap
            'target_img': torch.from_numpy(target_img).type(torch.FloatTensor),  # target-->areamap;
            'centermap_s': torch.from_numpy(centermap_source).type(torch.FloatTensor),
            'centermap_t': torch.from_numpy(centermap_target).type(torch.FloatTensor)}


class Dataset_contrast_reg(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

        # 读fake_img标签
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_target = csv.reader(ff)
            self.labels_target = list(reader_target)
            print('len', len(self.labels_target))
            self.namelist_target = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            target_sort = 1
            target_num = len(self.labels_target)

            while target_sort < target_num:
                self.namelist_target.append(target_sort)
                target_sort = target_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_target)} fake examples')
    def __len__(self):
        return max(len(self.labels_source) - 2, len(self.labels_target) - 2)

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
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        sourceimg_name = self.imgs_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img0, 1)
        # 读取source标签
        points_all = np.zeros([1, self.num_points * 2])  # 存放当前图中的所有人的所有关键点
        index = idx
        for n in range(self.num_points):
            numList = [m.start() for m in re.finditer('_', self.labels_source[index][n + 3])]
            point = [int(self.labels_source[index][n + 3][0:numList[0]]) / w - 1 / 2,
                     int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]]) / h - 1 / 2]  # 归一化的label点坐标(-0.5~0.5)
            points_all[0, 2 * n] = point[0]
            points_all[0, 2 * n + 1] = point[1]
        points_all = points_all[0]

        # 读取target imgs并生成位置热力图
        # if i >= len(self.labels_target) - 2:
        #     ii = random.sample(range(len(self.labels_target) - 2), 1)[0]
        # else:
        #     ii = i
        sort = random.sample(range(len(self.labels_target) - 2), 1)[0]
        idx = int(self.namelist_target[sort])
        people_num_t = int(self.labels_target[idx][1])  # 当前图像中的人数
        targetimg_name = self.imgs_dir + self.labels_target[idx][0]  # 获取图像名，读图
        # print('img:', sourceimg_name, targetimg_name)
        target_img = Image.open(targetimg_name)
        target_img0 = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)
        ht, wt, _ = target_img0.shape
        target_img = self.preprocess(self.resize_w, self.resize_h, target_img0, 1)
        # 根据关键点标签生成中心点热力图
        points_all_target = np.zeros([people_num_t, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num_t):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_target[index][n + 3])]
                point = [int(self.labels_target[index][n + 3][0:numList[0]]),
                         int(self.labels_target[index][n + 3][numList[0] + 1:numList[1]])]
                points_all_target[k, n, :] = point

        center = np.mean(points_all_target, axis=1)
        center = center[0]
        center[0] = center[0] / wt - 1 / 2
        center[1] = center[1] / ht - 1 / 2

        return {
            'source_img': torch.from_numpy(source_img).type(torch.FloatTensor),  # source-->heatmap
            'target_img': torch.from_numpy(target_img).type(torch.FloatTensor),  # target-->areamap;
            'points_all': torch.from_numpy(points_all).type(torch.FloatTensor),
            'center': torch.from_numpy(center).type(torch.FloatTensor)}


class Dataset_pose(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, label, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.label = label
        self.scale = scale
        self.num_points = num_points
        # 读img标签
        with open(self.label, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            print('len', len(self.labels))
            self.namelist = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:
                self.namelist.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist)} examples')

    def __len__(self):
        return len(self.namelist)

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
        # 读取images并生成关键点热力图
        if i >= len(self.labels) - 2:
            ii = random.sample(range(1, len(self.labels) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist[ii])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        numList0 = [m.start() for m in re.finditer("/", self.labels[idx][0])]
        folder_img = self.labels[idx][0][numList0[0] + 1 : numList0[1]]
        if folder_img == 'Rat1Crops_v2':
            sigma = 3
        else:
            sigma = 2
        # sigma = 3

        img_name = self.imgs_dir + self.labels[idx][0]
        img = Image.open(img_name)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)
        # 根据标签生成关键点热力图
        heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap
        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        # print('shape:', heatmaps.shape, img.shape)
        heatmaps = np.array(heatmaps)

        show_source = np.zeros([self.resize_h, self.resize_w, 3])
        for kk in range(3):
            show_source[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3

        cv2.imshow('show_source', show_source)
        cv2.waitKey(0)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),  # source-->heatmap
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)}


class Dataset_all_center(Dataset):
    def __init__(self, resize_w, resize_h, dir_img, source_label, target_label, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = dir_img
        self.source_label = source_label
        self.target_label = target_label
        self.scale = scale
        self.num_points = num_points
        # 读source_img标签
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader_source = csv.reader(f)
            self.labels_source = list(reader_source)
            print('len', len(self.labels_source))
            self.namelist_source = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels_source)

            while img_sort < trainset_num:
                self.namelist_source.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.namelist_source)} true examples')

    def __len__(self):
        return len(self.namelist_source)

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
        if img_nd.max() > 1:  # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        # 读取source images并生成关键点热力图
        if i >= len(self.labels_source) - 2:
            ii = random.sample(range(1, len(self.labels_source) - 2), 1)[0]
        else:
            ii = i + 1
        idx = int(self.namelist_source[ii])
        people_num = int(self.labels_source[idx][1])  # 当前图像中的人数
        sourceimg_name = self.imgs_dir + self.labels_source[idx][0]
        source_img = Image.open(sourceimg_name)
        source_img0 = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_RGB2BGR)
        h, w, _ = source_img0.shape
        source_img = self.preprocess(self.resize_w, self.resize_h, source_img0, 1)

        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer("_", self.labels_source[index][n + 3])]
                point = [int(self.labels_source[index][n + 3][0:numList[0]]),
                         int(self.labels_source[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point

        center_source = np.mean(points_all, axis=1)
        center_source = center_source[0]
        sigma = 3
        centermap0_source = CenterLabelHeatMapResize(h, w, center_source[0], center_source[1],
                                                     self.resize_h, self.resize_w, sigma)
        centermap_source = cv2.resize(centermap0_source, (28, 28))
        centermap_source = np.expand_dims(centermap_source, axis=0)
        centermap_source = np.array(centermap_source / 255)

        # debug_可视化标签
        # print(source_img0.shape, centermap0_source.shape, source_img.shape)
        # source_show = (cv2.resize(source_img0, (self.resize_w, self.resize_h))) * 0.5
        # source_show[:, :, 1] = source_show[:, :, 1] + centermap0_source

        # cv2.imshow('source:', np.uint8(source_show))
        # cv2.waitKey(0)

        return {
            'image': torch.from_numpy(source_img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(centermap_source).type(torch.FloatTensor)}   # centermap
