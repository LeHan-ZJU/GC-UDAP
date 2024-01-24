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


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
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


def CenterLabelHeatMapResize(img_height, img_width, c_x, c_y, resize_h, resize_w, sigma):
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

        # read labels
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = [] 
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:
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
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
            img_nd = img_nd.transpose((2, 0, 1))  # HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])
        img_file = self.imgs_dir + self.labels[idx][0]

        img = Image.open(img_file)
        RandomRate = random.sample(range(2), 1)
        if RandomRate[0] == 1:
            img1 = transforms.ColorJitter(brightness=0.5)(img)
            img = transforms.ColorJitter(contrast=1)(img1)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)
        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point
                sigma = 3

                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  #,
        }


class Dataset_DAT(Dataset):
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
        # source
        with open(self.source_label, 'r', encoding='gb18030') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.labels = self.labels[1:]
            print('len', len(self.labels))
            trainset_num = len(self.labels)
            logging.info(f'Creating dataset with {trainset_num} source examples')

        # target data with peusdo label
        with open(self.target_label, 'r', encoding='gb18030') as ff:
            reader_fake = csv.reader(ff)
            self.target_label = list(reader_fake)
            self.target_label = self.target_label[1:]
            print('len_target_label', len(self.target_label))
            Target_num = len(self.target_label)
            logging.info(f'Creating dataset with {Target_num} target examples with peusdo label')

        # target data without peusdo label
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
            img_nd = img_nd.transpose((2, 0, 1))  # HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        searchContext = "_"
        # target images
        if i >= len(self.target_label) - 2:
            idx = random.sample(range(1, len(self.target_label) - 2), 1)[0]
        else:
            idx = i + 1
        box = self.target_label[idx][2]
        # Determine the existence of pseudo-labels.
        numList = [m.start() for m in re.finditer(searchContext, box)]
        box = [int(box[0:numList[0]]), int(box[numList[0] + 1:numList[1]]),
               int(box[numList[1] + 1:numList[2]]), int(box[numList[2] + 1:])]
        if max(box) == 0:
            # If there are no pseudo-labels available, the image is randomly augmented, and then self-supervised training is conducted based on Graph Constraints.
            SelfSupervision = np.ones(1)
            img_file = self.imgs_dir + self.target_label[idx][0]
            img0 = Image.open(img_file)
            img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
            h, w, _ = img.shape
            img = AugImg(img, [self.resize_h, self.resize_w])
            img = self.preprocess(self.resize_w, self.resize_h, img, 1)

            target = transforms.ColorJitter(brightness=0.5)(img0)
            target = transforms.ColorJitter(contrast=1)(target)
            target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)

            target0 = rotate_img(target0, self.angle, [w / 2, h / 2])
            target = AugImg(target0, [self.resize_h, self.resize_w])
            target = self.preprocess(self.resize_w, self.resize_h, target, 1)

            heatmaps = np.zeros([self.num_points, int(self.resize_h / self.scale), int(self.resize_w / self.scale)])
            heatmaps_t = np.zeros([self.num_points, int(self.resize_h / self.scale), int(self.resize_w / self.scale)])

        else:
            SelfSupervision = np.zeros(1)
            img_name = self.target_label[idx][0]
            numList = [m.start() for m in re.finditer('/', img_name)]
            img_file = self.imgs_dir + self.target_label[idx][0]
            if img_file[-9:-5] == 'flip':
                if img_name[1: numList[1]] in (['maze', 'minefield', 'treadmill']):
                    target = Image.open(img_file[:-10] + '.bmp')
                else:
                    target = Image.open(img_file[:-10] + '.jpg')
                RandomRate = random.sample(range(3), 1)
                if RandomRate[0] == 1:
                    target1 = transforms.ColorJitter(brightness=0.5)(target)
                    target = transforms.ColorJitter(contrast=1)(target1)
                target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
                h, w, _ = target0.shape
                target0 = cv2.flip(cv2.flip(target0, 1), 0)

            else:
                target = Image.open(img_file)
                RandomRate = random.sample(range(3), 1) 
                if RandomRate[0] == 1:
                    target1 = transforms.ColorJitter(brightness=0.5)(target)
                    target = transforms.ColorJitter(contrast=1)(target1)
                target0 = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
                h, w, _ = target0.shape
            target = AugImg(target0, [self.resize_h, self.resize_w])
            target = self.preprocess(self.resize_w, self.resize_h, target, 1)

            # heatmap init
            heatmaps_t = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
            points_all_t = np.zeros([1, self.num_points, 2])
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.target_label[index][n + 3])]
                point_t = [int(self.target_label[index][n + 3][0:numList[0]]),
                           int(self.target_label[index][n + 3][numList[0] + 1:numList[1]])]
                point_t = Trans_point(point_t, h, w, [self.resize_h, self.resize_w])
                points_all_t[0, n, :] = point_t
                heatmap0_t = CenterLabelHeatMap(self.resize_h, self.resize_w, point_t[0], point_t[1], sigma=3)
                heatmap_t = cv2.resize(heatmap0_t, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps_t[:, :, n] = heatmaps_t[:, :, n] + heatmap_t

            # source imgs
            if i >= len(self.labels) - 2:
                idx = random.sample(range(len(self.labels) - 2), 1)[0]
            else:
                idx = i
            img_file = self.imgs_dir + self.labels[idx][0]
            img = Image.open(img_file)
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape
            img = AugImg(img0, [self.resize_h, self.resize_w])
            img = self.preprocess(self.resize_w, self.resize_h, img, 1)

            heatmaps = np.zeros([int(self.resize_h / self.scale), int(self.resize_w / self.scale), self.num_points])
            points_all = np.zeros([1, self.num_points, 2])
            index = idx
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[0, n, :] = point
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma=3)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

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
