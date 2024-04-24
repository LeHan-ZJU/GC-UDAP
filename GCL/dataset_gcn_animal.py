import re
import cv2
import csv
import torch
import random
import numpy as np
from PIL import Image
from torch_geometric.data import Data
from Tools.tools_graph import find_length, find_neighbor, cal_Angle, show_img, aug_data


def read_edge_index(num_points, s=0):
    if num_points == 6:
        if s == 0:  # Non-siamese network structure
            edge_index = [[0, 3], [3, 0], [0, 4], [4, 0],     # 3 neighbor
                          [1, 2], [2, 1], [1, 4], [4, 1],
                          [2, 5], [5, 2], [3, 5], [5, 3],
                          [0, 2], [2, 0], [1, 3], [3, 1], [4, 5], [5, 4],
                          [6, 9], [9, 6], [6, 10], [10, 6],
                          [7, 8], [8, 7], [7, 10], [10, 7],
                          [8, 11], [11, 8], [9, 11], [11, 9],
                          [6, 8], [8, 6], [7, 9], [9, 7], [10, 11], [10, 11]]
    elif num_points == 10:
        if s == 0:  
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
    elif num_points == 17:
        if s == 0:
            edge_index = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 5], [5, 0],
                          [1, 2], [2, 1], [1, 8], [8, 1], [2, 3], [3, 2],
                          [3, 4], [4, 3], [3, 5], [5, 3], [3, 8], [8, 3], 
                          [4, 11], [11, 4], [4, 14], [14, 4], 
                          [5, 6], [6, 5], [5, 8], [8, 5],
                          [6, 7], [7, 6], [6, 9], [9, 6],
                          [7, 10], [10, 7], [7, 13], [13, 7],
                          [8, 9], [9, 8], [9, 10], [10, 9], 
                          [10, 16], [16, 10], [11, 12], [12, 11], [11, 14], [14, 11],
                          [12, 13], [13, 12], [12, 15], [15, 12], 
                          [13, 16], [16, 13], [14, 15], [15, 14], [15, 16], [16, 15],
                          [17, 18], [18, 17], [17, 19], [19, 17], [17, 22], [22, 17],
                          [18, 19], [19, 18], [18, 25], [25, 18], [19, 20], [20, 19],
                          [20, 21], [21, 20], [20, 22], [22, 20], [20, 25], [25, 20], 
                          [21, 28], [28, 21], [21, 31], [31, 21], 
                          [22, 23], [23, 22], [22, 25], [25, 22],
                          [23, 24], [24, 23], [23, 26], [26, 23],
                          [24, 27], [27, 24], [24, 30], [30, 24],
                          [25, 26], [26, 25], [26, 27], [27, 26], 
                          [27, 33], [33, 27], [28, 29], [29, 28], [28, 31], [31, 28],
                          [29, 30], [30, 29], [29, 32], [32, 29], 
                          [30, 33], [33, 30], [31, 32], [32, 31], [32, 33], [33, 32]
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
            # Find the two adjacent points according to the edge index
            neighbors = find_neighbor(p, edge_index)
            if len(neighbors) >= 2:
                s = 0
                for v1 in range(len(neighbors) - 1):
                    for v2 in range(v1 + 1, len(neighbors)):
                        if max(AllPoints[0:2, neighbors[v1]]) == 0 and min(AllPoints[0:2, neighbors[v2]]) > 0:
                            angle[s] = 0  # When p is nonempty and either v1 or v2 is [0, 0] --> a=0
                        elif max(AllPoints[0:2, neighbors[v2]]) == 0 and min(AllPoints[0:2, neighbors[v1]]) > 0:
                            angle[s] = 0
                        elif max(AllPoints[0:2, neighbors[v1]]) == 0 and max(AllPoints[0:2, neighbors[v2]]) == 0:
                            angle[s] = -1    # When p is nonempty and v1 and v2 are both [0, 0] --> a=-1
                        else:
                            vector1 = AllPoints[0:2, neighbors[v1]] - point0
                            vector2 = AllPoints[0:2, neighbors[v2]] - point0
                            angle[s] = cal_Angle(vector1, vector2)
                        s = s + 1
        else:
            angle = -2 * angle  # p:[0, 0] --> a=-2
        angle = np.array(angle)

        angles.append(angle)
    angles = np.array(angles)

    return angles


def read_data(labels, imgs_dir, num_points, index1):
    img_name = imgs_dir + labels[index1][0]
    img = Image.open(img_name)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    numList = [m.start() for m in re.finditer("_", labels[index1][2])]  # read the box label to obtain the object length to determine the range of the disturbance noise
    box = [int(labels[index1][2][0:numList[0]]),
           int(labels[index1][2][numList[0] + 1:numList[1]]),
           int(labels[index1][2][numList[1] + 1:numList[2]]),
           int(labels[index1][2][numList[2] + 1:])]
    length = find_length(box[0:2], box[2:4])

    points_all = np.zeros([2, num_points])
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


def gcn_dataset_animal(dir_label, dir_img, num_points, noise_p, noise_n):
    data_list =[]
    # read csv
    with open(dir_label, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        labels = list(reader)
        print('len', len(labels))
        trainset_num = len(labels)

    edge_index = read_edge_index(num_points)
    for i in range(1, trainset_num):
        index1 = i
        x = np.zeros([8, num_points * 2])

        height, width, length, points1, img1 = read_data(labels, dir_img, num_points, index1)
        points1 = gen_sample_sNoise(points1, width, height, length, rate_p=noise_p)

        points2, img2 = aug_data(points1, img1, 180)

        postive = random.sample(range(3), 1)[0]
        if postive == 1:
            points2 = gen_sample_sNoise(points2, width, height, length, rate_p=noise_p)

        else:
            postive = 0
            case = random.sample(range(3), 1)[0]
            if case == 0:    # case=0:p1 and p2 adds the same noise to construct hard negative sample pairs
                points1 = gen_sample_bNoise(points1, width, height, length, rate_p=noise_p, rate_n=noise_n)
                points2, img2 = aug_data(points1, img1, 180)
            elif case == 1:  # case=1：p1, p2 add different large noises
                points1 = gen_sample_bNoise(points1, width, height, length, rate_p=noise_p, rate_n=noise_n)
                points2 = gen_sample_bNoise(points2, width, height, length, rate_p=noise_p, rate_n=noise_n)
            else:            # case=2：random(1p & 1n)
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
