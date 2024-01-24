import torch
import torch.nn as nn
from torchvision import models
from Models.do_conv_pytorch import DOConv2d
from transformer.ca import Multi_Attention2


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

        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv.weight, std=0.001)
        nn.init.constant_(self.outConv.bias, 0)

    def forward(self, x):
        f1 = self.Up1(x)
        f2 = self.Up2(f1)
        f3 = self.Up3(f2)
        out = self.outConv(f3)
        return out, f1


class FeatureAlignment(nn.Module):
    def __init__(self, args, attention_channel):
        super(FeatureAlignment, self).__init__()
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


class ContrastNet1_MultiA(nn.Module):
    def __init__(self, args, extract_list, device, nof_joints, train):
        super(ContrastNet1_MultiA, self).__init__()
        self.n_classes = nof_joints

        self.resnet = models.resnet50(pretrained=False)

        self.SubResnet = FeatureExtractor_v2(self.resnet, extract_list, attention_channel=args.d_model)
        self.FeatureAlignment = FeatureAlignment(args=args, attention_channel=args.d_model)
        self.PoseHead = PoseHead(attention_channel=args.d_model, nof_joints=self.n_classes)
        if train:
            self.SubResnet.load_state_dict(torch.load(args.path_backbone, map_location=device))
            print('Pretrained SubResnet weights have been loaded!')
            self.FeatureAlignment.load_state_dict(torch.load(args.path_transformer, map_location=device))
            print('Pretrained FeatureAlignment weights have been loaded!')
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
    def __init__(self, args, extract_list, device, nof_joints):
        super(ResSelf_pre, self).__init__()
        self.n_classes = nof_joints

        self.resnet = models.resnet50(pretrained=True)
        # if train:
        #     self.resnet.load_state_dict(torch.load(args.path_backbone, map_location=device))
        #     print('The pretrained weight of resnet50 has been loaded!')
        self.SubResnet = FeatureExtractor_v2(self.resnet, extract_list, attention_channel=args.d_model)

        self.FeatureAlignment = FeatureAlignment(args=args, attention_channel=args.d_model)
        self.PoseHead = PoseHead(attention_channel=args.d_model, nof_joints=self.n_classes)

    def forward(self, x):
        f0 = self.SubResnet(x)
        f, f2 = self.FeatureAlignment(f0, f0)
        out = self.PoseHead(f)
        return out
