import torch
import torch.nn as nn
from tqdm import tqdm

from GCN.graph_inference_block import heatmap_to_points, graph_inference


def eval_net_contrast(net, loader, Device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = nn.MSELoss()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            img = batch['image_s']
            img_t = batch['image_t']
            true_heatmaps = batch['heatmap_s']
            true_heatmaps_t = batch['heatmap_t']

            img = img.to(device=Device, dtype=torch.float32)
            img_t = img_t.to(device=Device, dtype=torch.float32)
            heatmap_type = torch.float32
            true_heatmaps = true_heatmaps.to(device=Device, dtype=heatmap_type)
            true_heatmaps_t = true_heatmaps_t.to(device=Device, dtype=heatmap_type)

            with torch.no_grad():
                pred1, pred2 = net(img, img_t)
                print('num_points:', pred2.shape[1], 'img_h:', img_t.shape[2], 'img_w:', img_t.shape[3])

            # 图推理筛选
            keyPoints = heatmap_to_points(pred2.numpy(),
                                          numPoints=pred2.shape[1],
                                          ori_W=img_t.shape[3],
                                          ori_H=img_t.shape[2])

            loss1 = criterion(pred1, true_heatmaps).item()
            loss2 = criterion(pred2, true_heatmaps_t).item()
            tot = loss1 + loss2
            pbar.update()

    net.train()
    return tot / n_val