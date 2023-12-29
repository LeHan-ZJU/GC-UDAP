import torch
import torch.nn as nn
from torchvision import models
from cross_model.cam import CAM
from Models.CBAM import CBAM
from Models.do_conv_pytorch import DOConv2d
from transformer.ca import Cross_Attention, Multi_Attention, Multi_Attention2, Multi_Attention2_noCA


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


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class FeatureExtractor_v2(nn.Module):
    def __init__(self, submodule, extracted_layers, attention_channel):
        super(FeatureExtractor_v2, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.conv = nn.Conv2d(2048, attention_channel, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        outputs = self.conv(outputs[0])
        return outputs


class PoseHead(nn.Module):
    def __init__(self, attention_channel, nof_joints):
        super(PoseHead, self).__init__()
        self.n_classes = nof_joints

        self.Up1 = up_conv(ch_in=attention_channel * 2, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv.weight, std=0.001)
        nn.init.constant_(self.outConv.bias, 0)

    def forward(self, x):
        f1 = self.Up1(x)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)
        return out, f1


class AttentionPart(nn.Module):
    def __init__(self, args, attention_channel):
        super(AttentionPart, self).__init__()
        self.attention_channel = attention_channel
        self.multi_a = Multi_Attention2(args=args, h=args.head, n=args.number, d_model=attention_channel,
                                        d_ff=args.d_ff, dropout=0.1)

    def forward(self, x1, x2):
        s = x1.shape
        # batch = s[0]

        # cross-attention # [b, 2048, 8, 10] - [b, 80, 1024]
        token_f1 = x1.view(s[0], self.attention_channel, s[2] * s[3])
        token_f2 = x2.view(s[0], self.attention_channel, s[2] * s[3])
        cmb_fea = torch.cat((token_f1, token_f2), dim=0)
        cmb_fea = cmb_fea.transpose(1, 2)
        ca_fea, sa_fea = self.multi_a(cmb_fea)  # [2b, 80, 1024]

        ca_fea_re = ca_fea.view(s[0] * 2, self.attention_channel, s[2], s[3])
        ca_fea_1 = ca_fea_re[:s[0]]
        ca_fea_2 = ca_fea_re[s[0]:]  # [b, 1048, 8, 10]

        sa_fea_re = sa_fea.view(s[0] * 2, self.attention_channel, s[2], s[3])
        sa_fea_1 = sa_fea_re[:s[0]]
        sa_fea_2 = sa_fea_re[s[0]:]  # [b, 1048, 8, 10]

        att_fea_1 = ca_fea_1 + sa_fea_1
        att_fea_2 = ca_fea_2 + sa_fea_2

        f1_cmb = torch.cat((att_fea_1, x1), dim=1)
        f2_cmb = torch.cat((att_fea_2, x2), dim=1)

        return f1_cmb, f2_cmb


class AttentionPart_noCA(nn.Module):
    def __init__(self, args, attention_channel):
        super(AttentionPart_noCA, self).__init__()
        self.attention_channel = attention_channel
        self.multi_a = Multi_Attention2_noCA(args=args, h=args.head, n=args.number, d_model=attention_channel,
                                             d_ff=args.d_ff, dropout=0.1)

    def forward(self, x1, x2):
        s = x1.shape
        # batch = s[0]

        # cross-attention # [b, 2048, 8, 10] - [b, 80, 1024]
        token_f1 = x1.view(s[0], self.attention_channel, s[2] * s[3])
        token_f2 = x2.view(s[0], self.attention_channel, s[2] * s[3])
        cmb_fea = torch.cat((token_f1, token_f2), dim=0)
        cmb_fea = cmb_fea.transpose(1, 2)
        ca_fea = self.multi_a(cmb_fea)  # [2b, 80, 1024]

        ca_fea_re = ca_fea.view(s[0] * 2, self.attention_channel, s[2], s[3])
        att_fea_1 = ca_fea_re[:s[0]]
        att_fea_2 = ca_fea_re[s[0]:]  # [b, 1048, 8, 10]

        f1_cmb = torch.cat((att_fea_1, x1), dim=1)
        f2_cmb = torch.cat((att_fea_2, x2), dim=1)

        return f1_cmb, f2_cmb



class ContrastNet(nn.Module):
    def __init__(self, model_path, extract_list, device, in_channel, nof_joints, train):
        super(ContrastNet, self).__init__()
        self.model_path = model_path
        self.device = device
        self.in_channel = in_channel
        self.extract_list = extract_list
        self.n_classes = nof_joints
        self.cam = CAM()

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        self.Up1_t = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40

        self.resnet = models.resnet50(pretrained=False)
        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            nn.init.normal_(self.outConvArea.weight, std=0.01)
            nn.init.constant_(self.outConvArea.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, x_s, x_t):
        f_s = self.SubResnet(x_s)
        f_t = self.SubResnet(x_t)
        # print('shape:', f_s[0].unsqueeze(1).shape, f_t[0].unsqueeze(1).shape)
        cam_s, cam_t = self.cam(f_s[0].unsqueeze(1), f_t[0].unsqueeze(1))
        # pose branch
        f1 = self.Up1(cam_s)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)

        #area branch
        f1_area = self.Up1_t(cam_t)
        out_area = self.outConvArea(f1_area)
        return out, out_area


class ResnetCam(nn.Module):
    def __init__(self, model_path, extract_list, device, train):
        super(ResnetCam, self).__init__()
        self.model_path = model_path
        self.device = device
        self.extract_list = extract_list
        self.cam = CAM()

        self.resnet = models.resnet50(pretrained=False)

        if train:
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, x_s, x_t):
        f_s = self.SubResnet(x_s)
        f_t = self.SubResnet(x_t)
        cam_s, cam_t = self.cam(f_s[0].unsqueeze(1), f_t[0].unsqueeze(1))
        return cam_s, cam_t


class ContrastNet0(nn.Module):
    def __init__(self, model_path, extract_list, device, in_channel, nof_joints, train):
        super(ContrastNet0, self).__init__()
        self.model_path = model_path
        self.device = device
        self.in_channel = in_channel
        self.extract_list = extract_list
        self.n_classes = nof_joints
        self.ResnetCam = ResnetCam(model_path, extract_list, device, train)

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        self.Up1_t = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            nn.init.normal_(self.outConvArea.weight, std=0.01)
            nn.init.constant_(self.outConvArea.bias, 0)

    def forward(self, x_s, x_t):
        cam_s, cam_t = self.ResnetCam(x_s, x_t)
        # pose branch
        f1 = self.Up1(cam_s)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)

        #area branch
        f1_area = self.Up1_t(cam_t)
        out_area = self.outConvArea(f1_area)
        return out, out_area


class ContrastNet_center(nn.Module):
    def __init__(self, model_path, extract_list, device, in_channel, nof_joints, train):
        super(ContrastNet_center, self).__init__()
        self.model_path = model_path
        self.device = device
        self.in_channel = in_channel
        self.extract_list = extract_list
        self.n_classes = nof_joints
        self.ResnetCam = ResnetCam(model_path, extract_list, device, train)

        self.Up1 = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        self.Up1_t = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outConvCenter = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            nn.init.normal_(self.outConvCenter.weight, std=0.01)
            nn.init.constant_(self.outConvCenter.bias, 0)

    def forward(self, x_s, x_t):
        cam_s, cam_t = self.ResnetCam(x_s, x_t)
        # pose branch
        f1 = self.Up1(cam_s)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)

        #center branch
        f1_area = self.Up1(cam_t)
        f2_area = self.Up2(f1_area)
        f3_area = self.Up3(f2_area)
        out_area = self.outConvCenter(f3_area)
        return out, out_area


class ContrastNet_center_ca(nn.Module):
    def __init__(self, args, model_path, extract_list, device, in_channel, nof_joints, train):
        super(ContrastNet_center_ca, self).__init__()
        self.model_path = model_path
        self.device = device
        self.in_channel = in_channel
        self.extract_list = extract_list
        self.n_classes = nof_joints
        self.ResnetCam = ResnetCam(model_path, extract_list, device, train)
        self.ca = Cross_Attention(args=args, h=8, n=1, d_model=2048, d_ff=1024, dropout=0.1)

        self.Up1 = up_conv(ch_in=4096, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        self.Up1_t = up_conv(ch_in=2048, ch_out=256)  # size: 16*20--32*40

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outConvCenter = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            nn.init.normal_(self.outConvCenter.weight, std=0.01)
            nn.init.constant_(self.outConvCenter.bias, 0)

    def forward(self, x_s, x_t):
        cam_s, cam_t = self.ResnetCam(x_s, x_t)
        batch = cam_s.size(0)
        c = cam_s.size(1)

        # cross-attention # [b, 2048, 8, 10] - [b, 80, 1024]
        token_fs = cam_s.view(batch, c, 8 * 10)
        token_ft = cam_t.view(batch, c, 8 * 10)
        cmb_fea = torch.cat((token_fs, token_ft), dim=0)
        cmb_fea = cmb_fea.transpose(1, 2)
        ca_fea = self.ca(cmb_fea)  # [2b, 80, 1024]
        ca_fea_re = ca_fea.view(batch * 2, 2048, 8, 10)

        ca_fea_s = ca_fea_re[:batch]
        ca_fea_t = ca_fea_re[batch:]  # [b, 1048, 8, 10]

        f_s_cmb = torch.cat((ca_fea_s, cam_s), dim=1)
        f_t_cmb = torch.cat((ca_fea_t, cam_t), dim=1)

        # pose branch
        f1 = self.Up1(f_s_cmb)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)

        #center branch
        f1_area = self.Up1(f_t_cmb)
        f2_area = self.Up2(f1_area)
        f3_area = self.Up3(f2_area)
        out_area = self.outConvCenter(f3_area)
        return out, out_area


class ContrastNet1_center_ca(nn.Module):
    def __init__(self, args, model_path, extract_list, device, in_channel, nof_joints, train):
        super(ContrastNet1_center_ca, self).__init__()
        self.model_path = model_path
        self.device = device
        self.in_channel = in_channel
        self.extract_list = extract_list
        self.n_classes = nof_joints
        self.resnet = models.resnet50(pretrained=False)
        self.ca = Cross_Attention(args=args, h=8, n=1, d_model=2048, d_ff=1024, dropout=0.1)

        self.Up1 = up_conv(ch_in=4096, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outConvCenter = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            nn.init.normal_(self.outConvCenter.weight, std=0.01)
            nn.init.constant_(self.outConvCenter.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, x_s, x_t):
        f_s = self.SubResnet(x_s)[0]
        f_t = self.SubResnet(x_t)[0]
        batch = f_s.size(0)
        c = f_s.size(1)

        # cross-attention # [b, 2048, 8, 10] - [b, 80, 1024]
        token_fs = f_s.view(batch, c, 8 * 10)
        token_ft = f_t.view(batch, c, 8 * 10)
        cmb_fea = torch.cat((token_fs, token_ft), dim=0)
        cmb_fea = cmb_fea.transpose(1, 2)
        ca_fea = self.ca(cmb_fea)  # [2b, 80, 1024]
        ca_fea_re = ca_fea.view(batch * 2, 2048, 8, 10)

        ca_fea_s = ca_fea_re[:batch]
        ca_fea_t = ca_fea_re[batch:]  # [b, 1048, 8, 10]

        f_s_cmb = torch.cat((ca_fea_s, f_s), dim=1)
        f_t_cmb = torch.cat((ca_fea_t, f_t), dim=1)

        # pose branch
        f1 = self.Up1(f_s_cmb)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)

        #center branch
        f1_area = self.Up1(f_t_cmb)
        f2_area = self.Up2(f1_area)
        f3_area = self.Up3(f2_area)
        out_area = self.outConvCenter(f3_area)
        return out, out_area


class ContrastNet1_ca(nn.Module):
    def __init__(self, args, model_path, extract_list, device, in_channel, nof_joints, train):
        super(ContrastNet1_ca, self).__init__()
        self.model_path = model_path
        self.device = device
        self.in_channel = in_channel
        self.extract_list = extract_list
        self.n_classes = nof_joints
        self.resnet = models.resnet50(pretrained=False)
        self.ca = Cross_Attention(args=args, h=8, n=1, d_model=2048, d_ff=1024, dropout=0.1)

        self.conv = nn.Conv2d(2048, 768, kernel_size=(1, 1), stride=(1, 1))

        self.Up1 = up_conv(ch_in=2816, ch_out=256)  # size: 16*20--32*40
        self.Up2 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80
        self.Up3 = up_conv(ch_in=256, ch_out=256)  # size: 32*40--64*80

        # 初始化输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outConvCenter = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            nn.init.normal_(self.outConvCenter.weight, std=0.01)
            nn.init.constant_(self.outConvCenter.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

    def forward(self, x_s, x_t):
        f_s = self.SubResnet(x_s)[0]
        f_t = self.SubResnet(x_t)[0]
        batch = f_s.size(0)

        # cross-attention # [b, 2048, 8, 10] - [b, 80, 1024]
        token_fs = f_s.view(batch, 768, 8 * 10)
        token_ft = f_t.view(batch, 768, 8 * 10)
        cmb_fea = torch.cat((token_fs, token_ft), dim=0)
        cmb_fea = cmb_fea.transpose(1, 2)
        ca_fea = self.ca(cmb_fea)  # [2b, 80, 1024]
        ca_fea_re = ca_fea.view(batch * 2, 768, 8, 10)

        ca_fea_s = ca_fea_re[:batch]
        ca_fea_t = ca_fea_re[batch:]  # [b, 1048, 8, 10]

        f_s_cmb = torch.cat((ca_fea_s, f_s), dim=1)
        f_t_cmb = torch.cat((ca_fea_t, f_t), dim=1)

        # pose branch
        f1 = self.Up1(f_s_cmb)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)

        #center branch
        f1_area = self.Up1(f_t_cmb)
        f2_area = self.Up2(f1_area)
        f3_area = self.Up3(f2_area)
        out_area = self.outConvCenter(f3_area)
        return out, out_area


class ContrastNet1_MultiA(nn.Module):
    def __init__(self, args, extract_list, device, nof_joints, train):
        super(ContrastNet1_MultiA, self).__init__()
        self.n_classes = nof_joints

        self.resnet = models.resnet50(pretrained=False)

        self.SubResnet = FeatureExtractor_v2(self.resnet, extract_list, attention_channel=args.d_model)
        self.TransformerPart = AttentionPart(args=args, attention_channel=args.d_model)
        self.PoseHead = PoseHead(attention_channel=args.d_model, nof_joints=self.n_classes)
        if train:
            self.SubResnet.load_state_dict(torch.load(args.path_backbone, map_location=device))
            print('Pretrained SubResnet weights have been loaded!')
            self.TransformerPart.load_state_dict(torch.load(args.path_transformer, map_location=device))
            print('Pretrained TransformerPart weights have been loaded!')
            self.PoseHead.load_state_dict(torch.load(args.path_head, map_location=device))
            print('Pretrained head weights have been loaded!')

    def forward(self, x1, x2):
        f10 = self.SubResnet(x1)
        f20 = self.SubResnet(x2)
        f1, f2 = self.TransformerPart(f10, f20)
        out1, out_f1 = self.PoseHead(f1)
        out2, out_f2 = self.PoseHead(f2)
        return out1, out2, f10, f20, out_f1, out_f2


class ResSelf_pre(nn.Module):
    def __init__(self, args, extract_list, device, nof_joints, train):
        super(ResSelf_pre, self).__init__()
        self.n_classes = nof_joints

        self.resnet = models.resnet50(pretrained=False)
        # if train:
        #     self.resnet.load_state_dict(torch.load(args.path_backbone, map_location=device))
        #     print('The pretrained weight of resnet50 has been loaded!')
        self.SubResnet = FeatureExtractor_v2(self.resnet, extract_list, attention_channel=args.d_model)

        self.TransformerPart = AttentionPart(args=args, attention_channel=args.d_model)
        self.PoseHead = PoseHead(attention_channel=args.d_model, nof_joints=self.n_classes)

    def forward(self, x):
        f0 = self.SubResnet(x)
        f, f2 = self.TransformerPart(f0, f0)
        out = self.PoseHead(f)
        return out


class ResSelfnoCA_pre(nn.Module):
    def __init__(self, args, extract_list, device, nof_joints, train):
        super(ResSelfnoCA_pre, self).__init__()
        self.n_classes = nof_joints

        self.resnet = models.resnet50(pretrained=False)
        if train:
            self.resnet.load_state_dict(torch.load(args.path_backbone))
            print('The pretrained weight of resnet50 has been loaded!')
        self.SubResnet = FeatureExtractor_v2(self.resnet, extract_list, attention_channel=args.d_model)

        self.TransformerPart = AttentionPart_noCA(args=args, attention_channel=args.d_model)
        self.PoseHead = PoseHead(attention_channel=args.d_model, nof_joints=self.n_classes)

    def forward(self, x):
        f = self.SubResnet(x)
        f, f2 = self.TransformerPart(f, f)
        out = self.PoseHead(f)
        return out