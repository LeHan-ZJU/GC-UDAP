import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from Models.do_conv_pytorch import DOConv2d
from cross_model.cam import CAM


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            DOConv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            DOConv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        # print('Channel:', x.shape, out.shape)
        return out


class Channel_Att(nn.Module):  # NAM注意力机制
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = DOConv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        # print('Spatial:', x.shape, out.shape)
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class TensorCat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, d):
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=d)
        return x


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class Trans_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Up_conv = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
        )

    def forward(self, x):
        return self.Up_conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DOConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Subnet_cam(nn.Module):
    def __init__(self, model_path, extract_list, device, train):
        super(Subnet_cam, self).__init__()
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device

        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        if train:
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层
        self.cam = CAM()

    def forward(self, img):
        f = self.SubResnet(img)
        f_cam, b = self.cam(f[0].unsqueeze(1), f[0].unsqueeze(1))
        return f_cam


class Net_cam(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(Net_cam, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.Subnet_cam = Subnet_cam(model_path, extract_list, device, train)

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)

    def forward(self, img):
        f_cam = self.Subnet_cam(img)
        # f_cam, b = self.cam(f[0].unsqueeze(1), f[0].unsqueeze(1))
        # print('f_cam:', f_cam.shape)
        f1 = self.Up1(f_cam)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)
        return out  #, f2[0]


# class Net_cam_mt(nn.Module):
#     def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
#         super(Net_cam_mt, self).__init__()
#         self.n_classes = nof_joints
#         self.n_channels = n_channels
#         self.model_path = model_path
#         self.extract_list = extract_list
#         self.device = device
#
#         self.cam = CAM()
#
#         self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
#         self.Up2 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
#         self.Up3 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
#
#         # 初始化输出层参数
#         self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
#
#         # 初始化定位输出层参数
#         self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
#
#         # 加载resnet
#         self.resnet = models.resnet50(pretrained=False)
#
#         if train:
#             nn.init.normal_(self.outConv.weight, std=0.001)
#             nn.init.constant_(self.outConv.bias, 0)
#             nn.init.normal_(self.outConvArea.weight, std=0.01)
#             nn.init.constant_(self.outConvArea.bias, 0)
#             self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
#         self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层
#
#     def forward(self, img):
#         f = self.SubResnet(img)
#         # print(f[0].shape)
#         f_cam, b = self.cam(f[0].unsqueeze(1), f[0].unsqueeze(1))
#         # print('f_cam:', f_cam.shape)
#         # f1 = self.Up1(f_cam)
#         f1 = self.Up1(f_cam)
#         out_area = self.outConvArea(f1)
#
#         # f2 = self.Up2(f1)
#         # f3 = self.Up3(f2)
#         # out = self.outConv(f3)
#         return out_area  # , out_area


class RatNet1_cam(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(RatNet1_cam, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device
        self.cam = CAM()

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        # 初始化定位输出层参数
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        # print(self.resnet)
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)

            nn.init.normal_(self.outConvArea.weight, std=0.01)
            nn.init.constant_(self.outConvArea.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f_cam, b = self.cam(f[0].unsqueeze(1), f[0].unsqueeze(1))
        # print('f_cam:', f_cam.shape)
        f1 = self.Up1(f_cam)
        out_area = self.outConvArea(f1)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)
        return out, out_area


class RatNet1(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints, trainGAN):
        super(RatNet1, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device
        self.trainGAN = trainGAN

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80

        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)

        # 初始化定位输出层参数
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

        if self.trainGAN == 1:
            if train:
                nn.init.normal_(self.outConvArea.weight, std=0.01)
                nn.init.constant_(self.outConvArea.bias, 0)
                self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        else:
            # 初始化输出层参数
            self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
            if train:
                nn.init.normal_(self.outConv.weight, std=0.001)
                nn.init.constant_(self.outConv.bias, 0)

                nn.init.normal_(self.outConvArea.weight, std=0.01)
                nn.init.constant_(self.outConvArea.bias, 0)
                # self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f1 = self.Up1(f[0])
        out_area = self.outConvArea(f1)
        if self.trainGAN == 1:
            return out_area
        else:
            f2 = self.Up2(f1)
            f3 = self.Up3(f2)
            out = self.outConv(f3)
            return out, out_area


class RatNet2_cam(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(RatNet2_cam, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device
        self.cam = CAM()

        self.UpConv2 = Trans_Conv(2048, 256)  # size: 16*20--32*40
        self.UpConv3 = Trans_Conv(256, 256)  # size: 32*40--64*80
        self.UpConv4 = Trans_Conv(256, 256)  # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        # 初始化定位输出层参数
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        # print(self.resnet)
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)

            nn.init.normal_(self.outConvArea.weight, std=0.01)
            nn.init.constant_(self.outConvArea.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f_cam, b = self.cam(f[0].unsqueeze(1), f[0].unsqueeze(1))
        # print('f_cam:', f_cam.shape)
        f1 = self.UpConv2(f_cam)
        out_area = self.outConvArea(f1)
        f2 = self.UpConv3(f1)
        f3 = self.UpConv4(f2)
        out = self.outConv(f3)
        return out, out_area


class RatNet2(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(RatNet2, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device

        self.UpConv2 = Trans_Conv(2048, 256)                # size: 16*20--32*40
        self.UpConv3 = Trans_Conv(256, 256)               # size: 32*40--64*80
        self.UpConv4 = Trans_Conv(256, 256)               # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        # 初始化定位输出层参数
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        # print(self.resnet)
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)

            nn.init.normal_(self.outConvArea.weight, std=0.01)
            nn.init.constant_(self.outConvArea.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f1 = self.UpConv2(f[0])
        out_area = self.outConvArea(f1)
        f2 = self.UpConv3(f1)
        f3 = self.UpConv4(f2)
        out = self.outConv(f3)
        return out, out_area


class PoseNet(nn.Module):
    def __init__(self, model_path, extract_list, device, train, nof_joints):
        super(PoseNet, self).__init__()
        self.n_classes = nof_joints
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80

        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f1 = self.Up1(f[0])
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)
        return out


class Net_ResnetAttention_DOConv(nn.Module):
    def __init__(self, model_path, extract_list, device, train, n_channels, nof_joints):
        super(Net_ResnetAttention_DOConv, self).__init__()
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.model_path = model_path
        self.extract_list = extract_list
        self.device = device

        # self.UpConv2 = Trans_Conv(2048, 256)                # size: 16*20--32*40
        # self.UpConv3 = Trans_Conv(256, 256)               # size: 32*40--64*80
        # self.UpConv4 = Trans_Conv(256, 256)               # size: 32*40--64*80
        # self.Conv1 = conv_block(ch_in=2048, ch_out=256)
        # self.Conv2 = conv_block(ch_in=256, ch_out=256)
        # self.Conv3 = conv_block(ch_in=256, ch_out=256)
        self.cbam0 = CBAM(channel=2048)
        self.cbam1 = CBAM(channel=256)
        self.cbam2 = CBAM(channel=256)
        self.cbam3 = CBAM(channel=256)
        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)   # size: 32*40--64*80


        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        # print(self.resnet)
        if train:
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, img):
        f = self.SubResnet(img)
        f1_0 = self.Up1(f[0])
        # print('f1:', f1_0.shape)
        # f1 = self.cbam1(f1) + f1
        f_cbam1 = self.cbam1(f1_0)
        # print('f:', f[0].shape, 'cbam1:', f_cbam1.shape)
        f1 = f_cbam1 + f1_0
        f2_0 = self.Up2(f1)
        f2 = self.cbam2(f2_0) + f2_0
        f3_0 = self.Up3(f2)
        f3 = self.cbam3(f3_0) + f3_0
        out = self.outConv(f3)
        return out  #, f[0]  # self.cbam0(f[0])

