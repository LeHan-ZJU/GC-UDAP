import cv2
import csv
import numpy as np


def read_labels(label_dir):
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
    return labels


def rotate_img(image, angle, center):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated_img


def Combine_csv(file1_path, file2_path, new_path):
    labels1 = read_labels(file1_path)
    labels2 = read_labels(file2_path)

    with open(new_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for l in range(len(labels1)):
            writer.writerow(labels1[l])
        for l in range(1, len(labels2)):
            writer.writerow(labels2[l])
