import numpy as np


def PCK_metric(pred, gt, sort1, sort2, percent):
    num_imgs, num_points, _ = pred.shape
    results = np.full((num_imgs, num_points), 0, dtype=np.float32)
    thrs = []

    for i in range(num_imgs):
        thr = cal_distance(gt[i, sort1, :], gt[i, sort2, :]) * percent
        thrs.append(thr)
        # thr = 20
        for j in range(num_points):
            distance = cal_distance(pred[i, j, :], gt[i, j, :])
            if distance <= thr:
                results[i, j] = 1

    thrs = np.array(thrs)
    print('mean:', np.mean(thrs))
    # 计算均值
    mean_points = np.mean(results, axis=0)  # 计算每个点的pck均值
    mean_all = np.mean(mean_points)         # 计算所有点的pck均值

    # 计算方差
    var_points = np.zeros([1, num_points])
    for k in range(num_points):             # 计算每个关键点的方差
        var_points[0, k] = np.var(results[:, k])

    results_reshape = results.reshape([1, num_imgs * num_points])
    var_all = np.var(results_reshape)       # 计算所有点的方差

    return mean_points, var_points, mean_all, var_all


def PCK_metric_box(pred, gt, sort1, sort2, percent):
    num_imgs, num_points, _ = pred.shape
    results = np.full((num_imgs, num_points), 0, dtype=np.float32)
    thrs = []

    for i in range(num_imgs):
        thr = find_length(gt[i, sort1, :], gt[i, sort2, :]) * percent
        thrs.append(thr)
        # thr = 20
        for j in range(num_points):
            if max(gt[i, j, :]) == 0:   # 判断标签中该点是否存在
                distance = 0
            else:
                distance = cal_distance(pred[i, j, :], gt[i, j, :])
            if distance <= thr:
                results[i, j] = 1

    # thrs = np.array(thrs)
    # print(thrs)
    # print('mean:', np.mean(thrs))
    # 计算均值
    mean_points = np.mean(results, axis=0)  # 计算每个点的pck均值
    mean_all = np.mean(mean_points)         # 计算所有点的pck均值

    # 计算方差
    var_points = np.zeros([1, num_points])
    for k in range(num_points):             # 计算每个关键点的方差
        var_points[0, k] = np.var(results[:, k])

    results_reshape = results.reshape([1, num_imgs * num_points])
    var_all = np.var(results_reshape)       # 计算所有点的方差

    return mean_points, var_points, mean_all, var_all


def PCK_metric_TigDog(pred, gt, sort1, sort2, percent):
    # 0-leftEye, 1-rightEye,
    # 2-chin,
    # 3-frontLeftHoof, 4-frontRightHoof, 5-backLeftHoof, 6-backRightHoof,
    # 7-tailStart,
    # 8-frontLeftKnee, 9-frontRightKnee, (elbow)
    # 10-backLeftKnee, 11-backRightKnee
    # 12-leftShoulder, 13-rightShoulder
    # 14-frontLeftHip, 15-frontRightHip,16-backLeftHip, 17-backRightHip
    # 18-neck
    # group: [0, 1], 2, [3,4,5,6], 7, [8,9], [10, 11], [12,13], [14, 15, 16, 17]
    keypoints = ['leftEye', 'rightEye', 'chin', 'frontLeftHoof', 'frontRightHoof', 'backLeftHoof', 'backRightHoof','tailStart',
                 'frontLeftKnee', 'frontRightKnee', 'backLeftKnee', 'backRightKnee', 'leftShoulder',
                 'rightShoulder', 'frontLeftHip', 'frontRightHip', 'backLeftHip', 'backRightHip', 'neck']
    # eye:[0,1], chin:[2], shoulder:[12, 13], hip:[16, 17], elbow:[14,15], knee:[8,9,10,11], hooves[3,4,5,6]

    group_list = [[0, 1], [2], [12, 13], [16, 17], [14, 15], [8, 9, 10, 11], [3, 4, 5, 6]]
    # print(len(group_list), len(group_list[0]))
    num_imgs, num_points, _ = pred.shape
    results_ori = np.full((num_imgs, num_points), 0, dtype=np.float32)
    results_combine = np.full((num_imgs, len(group_list)), 0, dtype=np.float32)
    thrs = []

    for i in range(num_imgs):
        thr = find_length(gt[i, sort1, :], gt[i, sort2, :]) * percent
        thrs.append(thr)
        # thr = 20
        for j in range(num_points-1):
            if max(gt[i, j, :]) == 0:   # 判断标签中该点是否存在
                distance = 0
            else:
                distance = cal_distance(pred[i, j, :], gt[i, j, :])
            if distance <= thr:
                results_ori[i, j] = 1
        for k in range(len(group_list)):
            for m in range(len(group_list[k])):
                results_combine[i, k] = results_combine[i, k] + results_ori[i, group_list[k][m]]
            results_combine[i, k] = results_combine[i, k] / len(group_list[k])
            # print(results_ori[i, :], results_combine[i, :])

    # thrs = np.array(thrs)
    # print(thrs)
    # print('mean:', np.mean(thrs))
    # 计算均值
    mean_points_ori = np.mean(results_ori, axis=0)  # 计算每个点的pck均值
    mean_all_ori = np.mean(mean_points_ori)         # 计算所有点的pck均值

    mean_points_combine = np.mean(results_combine, axis=0)  # 计算每个点的pck均值
    mean_all_combine = np.mean(mean_points_combine)  # 计算所有点的pck均值

    return mean_points_combine, mean_points_ori, mean_all_combine, mean_all_ori


def cal_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[0]-p2[0])**2)


def find_length(p1, p2):
    l1 = np.abs(p1[0] - p2[0])
    l2 = np.abs(p1[1] - p2[1])
    return max(l1, l2)


if __name__ == '__main__':
    # gt_file = "E:/Codes/Mine/RatPose_paper/results_location/Debug1/points_all_gt.npy"
    # pred_file = "E:/Codes/Mine/RatPose_paper/results_location/Debug1/points_all_pred.npy"
    gt_file = "E:/Codes/Mine/RatPose_paper/results_coco/Debug_ResnetADOConv_crop/2stage/points_all_gt.npy"
    pred_file = "E:/Codes/Mine/RatPose_paper/results_coco/Debug_ResnetADOConv_crop/2stage/points_all_pred.npy"

    pred_data = np.load(pred_file)
    gt_data = np.load(gt_file)
    print(pred_data.shape, gt_data.shape)
    mean_points, var_points, mean_all, var_all = PCK_metric_box(pred_data, gt_data, 4, 5, 0.3)
    print('pck_points_mean:', mean_points)
    print('pck_points_val:', var_points)
    print('pck_all_mean:', mean_all, '    pck_all_val:', var_all)