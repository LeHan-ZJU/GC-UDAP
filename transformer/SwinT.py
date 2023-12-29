
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import swin_transformer


class HeatmapSwinTransformer(nn.Module):
    def __init__(self, num_joints, pretrain_weight, device, feature_size=96):
        super(HeatmapSwinTransformer, self).__init__()

        self.backbone = swin_transformer.SwinTransformer(
            img_size=384,
            in_chans=3,
            # num_classes=None,
            embed_dim=feature_size,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            ape=False,
            patch_norm=True,
            use_checkpoint=False)

        # pretrained_weights = torch.load(pretrain_weight, map_location='cpu')['model']
        # self.backbone.load_state_dict(pretrained_weights)
        self.backbone.load_state_dict(torch.load(pretrain_weight, map_location=device), strict=False)

        # 计算热力图的卷积层
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, num_joints, kernel_size=1)
        )

    def forward(self, x):
        # 前向计算
        x = self.backbone.forward_features(x)
        print('x_backbone:', x.shape)
        x = self.heatmap_conv(x)
        print('x_up:', x.shape)
        x = F.interpolate(x, scale_factor=4., mode='bilinear', align_corners=False)
        print('x_out:', x.shape)

        return x
