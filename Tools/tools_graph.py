import cv2
import math
import random
import numpy as np
from Tools.process import rotate_img


def rotate_coordinates(p, angle, center):
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
    # Randomly augment the input image

    h, w, _ = img.shape
    rot_center = [w / 2, h / 2]
    rot_points, rot_angle, rot_center = random_aug_v2(points, a_rotate, rot_center)
    rot_img = rotate_img(img, rot_angle, rot_center)
    return rot_points, rot_img


def random_aug_v2(p1, a_rotate, center):
    # Rotate with the image center point as the origin
    coor_r = np.zeros((2, p1.shape[1]))

    for i in range(p1.shape[1]):
        if max(p1[:, i]) == 0:
            coor_r[:, i] = p1[:, i]
        else:
            coor_r[:, i] = rotate_coordinates(p1[:, i], a_rotate, center)

    return coor_r, a_rotate, center


def find_length(p1, p2):  
    # Calculate the distance between two points x, y and return the larger one
    l1 = int(np.abs(p1[0] - p2[0]))
    l2 = int(np.abs(p1[1] - p2[1]))
    return max(l1, l2)


def cal_Angle(v1, v2):
    # Calculate the Angle between two vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    m = np.dot(v1, v2)
    cos_ = m / (norm1 * norm2)
    inv = np.arccos(cos_) * 180 / np.pi

    if np.isnan(inv):
        inv = 0

    return inv
