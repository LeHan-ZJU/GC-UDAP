import torch
import torch.nn as nn
from .sa import Self_Attention
from .ca import Cross_Attention
from .rn import Relation_Network, cos_similar


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


class SeqConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, mid_channels, out_channels, up=True):
        super().__init__()
        # mid_channels = out_channels * 2
        if up:
            self.Up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )
        else:
            self.Up_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

    def forward(self, x):
        return self.Up_conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # DOConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class TransformerPart(nn.Module):
    def __init__(self, args):
        super(TransformerPart, self).__init__()

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)

    def forward(self, img):
        sa_fea, left_tokens, idxs = self.sa(img)  # [b, 197, 768]
        ca_fea = self.ca(sa_fea)  # [b, 197, 768]

        # cls_fea = ca_fea[:, 0]  # [b, 1, 768]
        token_fea = ca_fea[:, 1:]  # [b, 196, 768]
        batch = token_fea.size(0)

        token_fea = token_fea.view(batch, 768, 14, 14)
        return token_fea


class Model_original(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.rn = Relation_Network(args.anchor_number, dropout=0.1)
        self.conv2d = nn.Conv2d(768, 512, 2, 2)


    def forward(self, sk, im, stage='train', only_sa=False):

        if stage == 'train':

            sk_im = torch.cat((sk, im), dim=0)
            sa_fea, left_tokens, idxs = self.sa(sk_im)  # [4b, 197, 768]
            ca_fea = self.ca(sa_fea)  # [4b, 197, 768]

            cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
            token_fea = ca_fea[:, 1:]  # [4b, 196, 768]
            batch = token_fea.size(0)

            token_fea = token_fea.view(batch, 768, 14, 14)
            down_fea = self.conv2d(token_fea)
            down_fea = down_fea.view(batch, 512, 7*7)
            down_fea = down_fea.transpose(1, 2)  # [4b, 49, 512]

            sk_fea = down_fea[:batch // 2]
            im_fea = down_fea[batch // 2:]
            cos_scores = cos_similar(sk_fea, im_fea)  # [2b, 49, 49]
            cos_scores = cos_scores.view(batch // 2, -1)
            rn_scores = self.rn(cos_scores)  # [2b, 1]

            # print('cls_fea:', cls_fea.size())
            # print('rn_scores:', cls_fea.size())
            return cls_fea, rn_scores

        else:

            if only_sa:
                sa_fea, left_tokens, idxs = self.sa(sk)  # [b, 197, 768]
                return sa_fea, idxs
            else:
                sk_im = torch.cat((sk, im), dim=0)
                ca_fea = self.ca(sk_im)  # [2b, 197, 768]

                cls_fea = ca_fea[:, 0]  # [2b, 1, 768]
                token_fea = ca_fea[:, 1:]  # [2b, 196, 768]
                batch = token_fea.size(0)

                token_fea = token_fea.view(batch, 768, 14, 14)
                down_fea = self.conv2d(token_fea)
                down_fea = down_fea.view(batch, 512, 7 * 7)
                down_fea = down_fea.transpose(1, 2)  # [2b, 49, 512]

                sk_fea = down_fea[:batch // 2]
                im_fea = down_fea[batch // 2:]
                cos_scores = cos_similar(sk_fea, im_fea)  # [b, 49, 49]
                cos_scores = cos_scores.view(batch // 2, -1)
                rn_scores = self.rn(cos_scores)  # [b, 49, 49]

                # print('cls_fea:', cls_fea.size())
                # print('rn_scores:', cls_fea.size())
                return cls_fea, rn_scores


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.conv2d = nn.Conv2d(768, 512, 2, 2)
        self.conv = SeqConv(768, 512, 256, up=True)

        # 初始化定位输出层参数
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConvArea.weight, std=0.001)
        nn.init.constant_(self.outConvArea.bias, 0)

    def forward(self, sk, im):
        sk_im = torch.cat((sk, im), dim=0)
        sa_fea, left_tokens, idxs = self.sa(sk_im)  # [4b, 197, 768]
        ca_fea = self.ca(sa_fea)  # [4b, 197, 768]

        # cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
        token_fea = ca_fea[:, 1:]  # [4b, 196, 768]
        batch = token_fea.size(0)

        token_fea = token_fea.view(batch, 768, 14, 14)
        up_fea = self.conv(token_fea)  # [4b, 768, 28, 28]

        sk_fea = up_fea[:batch // 2]
        im_fea = up_fea[batch // 2:]  # [2b, 768, 28, 28]

        sk_out = self.outConvArea(sk_fea)
        im_out = self.outConvArea(im_fea)

        return sk_out, im_out


class Model_pose_v0(nn.Module):
    def __init__(self, args):
        super(Model_pose_v0, self).__init__()

        self.args = args
        self.n_classes = args.num_points

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        # self.conv = SeqConv(768, 512, 256, up=True)
        self.Up1 = Trans_Conv(768, 256)  # size: 14*14--28*28
        self.Up2s = Trans_Conv(256, 256)  # size: 28*28--56*56
        self.Up3s = Trans_Conv(256, 256)  # size: 56*56--112*112
        self.Up2t = Trans_Conv(256, 256)  # size: 28*28--56*56
        self.Up3t = Trans_Conv(256, 256)  # size: 56*56--112*112

        # 初始化定位输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv.weight, std=0.001)
        nn.init.constant_(self.outConv.bias, 0)
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConvArea.weight, std=0.001)
        nn.init.constant_(self.outConvArea.bias, 0)

    def forward(self, sk, im):
        sk_im = torch.cat((sk, im), dim=0)
        sa_fea, left_tokens, idxs = self.sa(sk_im)  # [4b, 197, 768]
        ca_fea = self.ca(sa_fea)  # [4b, 197, 768]

        # cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
        token_fea = ca_fea[:, 1:]  # [4b, 196, 768]
        batch = token_fea.size(0)

        token_fea = token_fea.view(batch, 768, 14, 14)
        up_fea = self.Up1(token_fea)  # [4b, 768, 28, 28]

        sk_fea1 = up_fea[:batch // 2]
        im_fea1 = up_fea[batch // 2:]  # [2b, 768, 28, 28]

        # pose branch
        sk_fea2 = self.Up2s(sk_fea1)
        sk_fea3 = self.Up3s(sk_fea2)
        sk_out = self.outConv(sk_fea3)

        # area branch
        im_fea2 = self.Up2t(im_fea1)
        im_fea3 = self.Up3t(im_fea2)
        im_out = self.outConvArea(im_fea3)

        return sk_out, im_out


class Model_pose_v1(nn.Module):
    def __init__(self, args):
        super(Model_pose_v1, self).__init__()

        self.args = args
        self.n_classes = args.num_points

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.rn = Relation_Network(args.anchor_number, dropout=0.1)
        # self.conv = SeqConv(768, 512, 256, up=True)
        self.Up1 = up_conv(768, 256)  # size: 14*14--28*28
        self.Up2 = up_conv(256, 256)  # size: 28*28--56*56
        self.Up3 = up_conv(256, 256)  # size: 56*56--112*112
        # self.Up2t = up_conv(256, 256)  # size: 28*28--56*56
        # self.Up3t = up_conv(256, 256)  # size: 56*56--112*112

        # 初始化定位输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv.weight, std=0.001)
        nn.init.constant_(self.outConv.bias, 0)
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConvArea.weight, std=0.001)
        nn.init.constant_(self.outConvArea.bias, 0)

    def forward(self, sk, im):
        sk_im = torch.cat((sk, im), dim=0)
        sa_fea, left_tokens, idxs = self.sa(sk_im)  # [4b, 197, 768]
        ca_fea = self.ca(sa_fea)  # [4b, 197, 768]

        # cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
        token_fea = ca_fea[:, 1:]  # [4b, 196, 768]
        batch = token_fea.size(0)

        token_fea = token_fea.view(batch, 768, 14, 14)
        up_fea = self.Up1(token_fea)  # [4b, 768, 28, 28]

        sk_fea1 = up_fea[:batch // 2]
        im_fea1 = up_fea[batch // 2:]  # [2b, 768, 28, 28]

        # pose branch
        sk_fea2 = self.Up2(sk_fea1)
        sk_fea3 = self.Up3(sk_fea2)
        sk_out = self.outConv(sk_fea3)

        # area branch
        im_fea2 = self.Up2(im_fea1)
        im_fea3 = self.Up3(im_fea2)
        im_out = self.outConvArea(im_fea3)

        return sk_out, im_out


class Model_pose_pre(nn.Module):
    def __init__(self, args):
        super(Model_pose_pre, self).__init__()

        self.args = args
        self.n_classes = args.num_points

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)

        self.Up1 = up_conv(768, 256)  # size: 14*14--28*28
        self.Up2 = up_conv(256, 256)  # size: 28*28--56*56
        self.Up3 = up_conv(256, 256)  # size: 56*56--112*112

        # 初始化定位输出层参数
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv.weight, std=0.001)
        nn.init.constant_(self.outConv.bias, 0)
        self.outConvArea = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConvArea.weight, std=0.001)
        nn.init.constant_(self.outConvArea.bias, 0)

    def forward(self, im1, im2):
        cmb_im = torch.cat((im1, im2), dim=0)
        sa_fea, left_tokens, idxs = self.sa(cmb_im)  # [4b, 197, 768]
        ca_fea = self.ca(sa_fea)  # [4b, 197, 768]
        # cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
        token_fea = ca_fea[:, 1:]  # [4b, 196, 768]
        batch = token_fea.size(0)
        token_fea = token_fea.view(batch, 768, 14, 14)
        up_fea = self.Up1(token_fea)  # [4b, 768, 28, 28]

        im1_fea1 = up_fea[:batch // 2]
        im2_fea1 = up_fea[batch // 2:]  # [2b, 768, 28, 28]

        # pose branch
        im1_fea2 = self.Up2(im1_fea1)
        im1_fea3 = self.Up3(im1_fea2)
        im1_out = self.outConv(im1_fea3)

        # pose  branch
        im2_fea2 = self.Up2(im2_fea1)
        im2_fea3 = self.Up3(im2_fea2)
        im2_out = self.outConv(im2_fea3)

        return im1_out, im2_out


class Model_pose_v4(nn.Module):
    def __init__(self, args):
        super(Model_pose_v4, self).__init__()

        self.args = args
        self.n_classes = args.num_points

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff,
                                  dropout=0.1)

        self.Up1 = up_conv(768 * 2, 512)  # size: 14*14--28*28
        self.Up2 = up_conv(512, 256)  # size: 28*28--56*56
        self.Up3 = up_conv(256, 256)  # size: 56*56--112*112

        # 初始化定位输出层参数
        self.outConv1 = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv1.weight, std=0.001)
        nn.init.constant_(self.outConv1.bias, 0)
        self.outConv2 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv2.weight, std=0.001)
        nn.init.constant_(self.outConv2.bias, 0)

    def forward(self, im1, im2):
        cmb_im = torch.cat((im1, im2), dim=0)
        sa_fea, left_tokens, idxs = self.sa(cmb_im)  # [2b, 197, 768]

        ca_fea = self.ca(sa_fea)  # [2b, 197, 768]

        token_sa_fea = sa_fea[:, 1:]  # [2b, 196, 768]
        batch = token_sa_fea.size(0)
        token_sa_fea = token_sa_fea.view(batch, 768, 14, 14)
        im1_sa_fea = token_sa_fea[:batch // 2]
        im2_sa_fea = token_sa_fea[batch // 2:]  # [b, 768, 14, 14]

        token_ca_fea = ca_fea[:, 1:]  # [2b, 196, 768]
        token_ca_fea = token_ca_fea.view(batch, 768, 14, 14)
        im1_ca_fea = token_ca_fea[:batch // 2]
        im2_ca_fea = token_ca_fea[batch // 2:]  # [b, 768, 14, 14]

        im1_fea = torch.cat((im1_sa_fea, im1_ca_fea), dim=1)
        im2_fea = torch.cat((im2_sa_fea, im2_ca_fea), dim=1)

        # pose branch
        im1_fea1 = self.Up1(im1_fea)  # [b, 768, 28, 28]
        im1_fea2 = self.Up2(im1_fea1)
        im1_fea3 = self.Up3(im1_fea2)
        im1_out = self.outConv1(im1_fea3)

        # pose branch
        im2_fea1 = self.Up1(im2_fea)  # [b, 768, 28, 28]
        im2_fea2 = self.Up2(im2_fea1)
        im2_fea3 = self.Up3(im2_fea2)
        im2_out = self.outConv2(im2_fea3)

        return im1_out, im2_out


class Model_pose_pre_v3(nn.Module):
    def __init__(self, args):
        super(Model_pose_pre_v3, self).__init__()

        self.args = args
        self.n_classes = args.num_points

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff,
                                  dropout=0.1)
        self.Up1_s = up_conv(768, 256)  # size: 14*14--28*28
        self.Up1_c = up_conv(768, 256)  # size: 14*14--28*28
        self.Up2 = up_conv(512, 256)  # size: 28*28--56*56
        self.Up3 = up_conv(256, 256)  # size: 56*56--112*112
        self.Up2_2 = up_conv(512, 256)  # size: 28*28--56*56
        self.Up3_2 = up_conv(256, 256)  # size: 56*56--112*112

        # 初始化定位输出层参数
        self.outConv1 = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv1.weight, std=0.001)
        nn.init.constant_(self.outConv1.bias, 0)
        self.outConv2 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv2.weight, std=0.001)
        nn.init.constant_(self.outConv2.bias, 0)

    def forward(self, im1, im2):
        cmb_im = torch.cat((im1, im2), dim=0)
        sa_fea, left_tokens, idxs = self.sa(cmb_im)  # [4b, 197, 768]
        token_sa_fea = sa_fea[:, 1:]  # [4b, 196, 768]
        batch = token_sa_fea.size(0)
        token_sa_fea = token_sa_fea.view(batch, 768, 14, 14)
        up_sa_fea = self.Up1_s(token_sa_fea)  # [4b, 768, 28, 28]
        im1_sa_fea1 = up_sa_fea[:batch // 2]
        im2_sa_fea1 = up_sa_fea[batch // 2:]  # [2b, 768, 28, 28]

        ca_fea = self.ca(sa_fea)  # [4b, 197, 768]
        token_ca_fea = ca_fea[:, 1:]  # [4b, 196, 768]
        batch = token_ca_fea.size(0)
        token_ca_fea = token_ca_fea.view(batch, 768, 14, 14)
        up_ca_fea = self.Up1_c(token_ca_fea)  # [4b, 768, 28, 28]
        im1_ca_fea1 = up_ca_fea[:batch // 2]
        im2_ca_fea1 = up_ca_fea[batch // 2:]  # [2b, 768, 28, 28]

        im1_fea1 = torch.cat((im1_sa_fea1, im1_ca_fea1), dim=1)
        im2_fea1 = torch.cat((im2_sa_fea1, im2_ca_fea1), dim=1)

        # pose branch
        im1_fea2 = self.Up2(im1_fea1)
        im1_fea3 = self.Up3(im1_fea2)
        im1_out = self.outConv1(im1_fea3)

        # pose branch
        im2_fea2 = self.Up2_2(im2_fea1)
        im2_fea3 = self.Up3_2(im2_fea2)
        im2_out = self.outConv2(im2_fea3)

        return im1_out, im2_out


class Model_pose_reg(nn.Module):
    def __init__(self, args):
        super(Model_pose_reg, self).__init__()

        self.args = args
        self.n_classes = args.num_points

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.conv2d = nn.Conv2d(768, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 初始化定位输出层参数
        self.outPose = nn.Linear(512, self.n_classes * 2)
        self.outCenter = nn.Linear(512, 2)

    def forward(self, sk, im):
        sk_im = torch.cat((sk, im), dim=0)
        sa_fea, left_tokens, idxs = self.sa(sk_im)  # [2b, 197, 768]
        ca_fea = self.ca(sa_fea)  # [2b, 197, 768]

        # cls_fea = ca_fea[:, 0]  # [2b, 1, 768]
        token_fea = ca_fea[:, 1:]  # [2b, 196, 768]
        batch = token_fea.size(0)

        token_fea = token_fea.view(batch, 768, 14, 14)
        down_fea = self.conv2d(token_fea)  # [2b, 768, 7, 7]
        down_fea = self.avgpool(down_fea)  # [2b, 768, 1, 1]

        sk_fea1 = down_fea[:batch // 2]
        sk_fea = torch.flatten(sk_fea1, 1)
        im_fea1 = down_fea[batch // 2:]  # [b, 768, 1, 1]
        im_fea = torch.flatten(im_fea1, 1)

        # print('fea:', im_fea.shape, sk_fea.shape)

        # pose branch
        sk_out = self.outPose(sk_fea)
        # area branch
        im_out = self.outCenter(im_fea)
        return sk_out, im_out


class SelfAtten_pose_pre(nn.Module):
    def __init__(self, args):
        super(SelfAtten_pose_pre, self).__init__()
        self.args = args
        self.n_classes = args.num_points
        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)

        self.Up1 = up_conv(768, 512)  # size: 14*14--28*28
        self.Up2 = up_conv(512, 256)  # size: 28*28--56*56
        self.Up3 = up_conv(256, 256)  # size: 56*56--112*112

        # 初始化定位输出层参数
        self.outConv1 = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv1.weight, std=0.001)
        nn.init.constant_(self.outConv1.bias, 0)

    def forward(self, im):
        sa_fea, left_tokens, idxs = self.sa(im)  # [b, 197, 768]
        token_sa_fea = sa_fea[:, 1:]  # [b, 196, 768]
        batch = token_sa_fea.size(0)
        token_sa_fea = token_sa_fea.view(batch, 768, 14, 14)

        up_sa_fea = self.Up1(token_sa_fea)  # [4b, 768, 28, 28]
        im_fea2 = self.Up2(up_sa_fea)
        im_fea3 = self.Up3(im_fea2)
        out = self.outConv1(im_fea3)
        return out


if __name__ == '__main__':
    # args = Option().parse()
    sk = torch.rand((4, 224, 224))
    im = torch.rand((4, 224, 224))
    # model = Model(args)
    # cls_fea, rn_scores = model(sk, im)
