import torch
import torch.nn as nn
from .sa_da import Self_Attention
from .ca import Cross_Attention


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
        sa_fea, left_tokens, idxs, Q, K, V = self.sa(im)  # [b, 197, 768]
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