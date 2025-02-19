import json
from operator import ne
import torch.nn as nn
import torch
import os
import sys
os.chdir(sys.path[0])
import torch.nn.functional as F
from timm.models import register_model

""""""
import json
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

"""
    逐层卷积
"""
class DepthwiseConv(nn.Module):
    """
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，元组类型
        padding: 补充
        stride: 步长
    """

    def __init__(self, in_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False):
        super(DepthwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x):
        out = self.conv(x)
        return out
"""
    逐点卷积
"""
class PointwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
"""
    深度可分离卷积
"""
class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(DepthwiseSeparableConv, self).__init__()

        self.conv1 = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.conv2 = PointwiseConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
"""
    下采样
    [batch_size, in_channels, height, width] -> [batch_size, out_channels, height // stride, width // stride]
"""
class DownSampling(nn.Module):
    """
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        norm_layer: 正则化层，如果为None，使用BatchNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[-1] // 2)
        )

        if norm_layer is None:
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.norm = norm_layer

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return out

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(
            self,
            args=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 100,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(_MatrixDecomposition2DBase, self).__init__()
        args: dict = json.loads(args)
        for k, v in args.items():
            setattr(self, k, v)

    @abstractmethod
    def _build_bases(self, batch_size):
        pass

    @abstractmethod
    def local_step(self, x, bases, coef):
        pass

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batchszie * MD_S, N, MD_R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.INV_T * coef, dim=-1)

        steps = self.TRAIN_STEPS if self.training else self.EVAL_STEPS
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    @abstractmethod
    def compute_coef(self, x, bases, coef):
        pass

    def forward(self, x):

        batch_size, channels, height, width = x.shape

        # (batch_size, channels, height, width) -> (batch_size * MD_S, MD_D, N)
        if self.SPATIAL:
            self.MD_D = channels // self.MD_S
            N = height * width
            x = x.view(batch_size * self.MD_S, self.MD_D, N)
        else:
            self.MD_D = height * width
            N = channels // self.MD_S
            x = x.view(batch_size * self.MD_S, N, self.MD_D).transpose(1, 2)

        if not self.RAND_INIT and not hasattr(self, 'bases'):
            bases = self._build_bases(1)
            self.register_buffer('bases', bases)

        # (MD_S, MD_D, MD_R) -> (batch_size * MD_S, MD_D, MD_R)
        if self.RAND_INIT:
            bases = self._build_bases(batch_size)
        else:
            bases = self.bases.repeat(batch_size, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (batch_size * MD_S, N, MD_R)
        coef = self.compute_coef(x, bases, coef)

        # (batch_size * MD_S, MD_D, MD_R) @ (batch_size * MD_S, N, MD_R)^T -> (batch_size * MD_S, MD_D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (batch_size * MD_S, MD_D, N) -> (batch_size, channels, height, width)
        if self.SPATIAL:
            x = x.view(batch_size, channels, height, width)
        else:
            x = x.transpose(1, 2).view(batch_size, channels, height, width)

        # (batch_size * height, MD_D, MD_R) -> (batch_size, height, N, MD_D)
        bases = bases.view(batch_size, self.MD_S, self.MD_D, self.MD_R)

        if self.return_bases:
            return x, bases
        return x

class NMF2D(_MatrixDecomposition2DBase):
    def __init__(
            self,
            args=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(NMF2D, self).__init__(args)

    def _build_bases(self, batch_size):
        bases = torch.rand((batch_size * self.MD_S, self.MD_D, self.MD_R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batch_size * MD_S, N, MD_R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (batch_size * MD_S, N, MD_R) @ [(batch_size * MD_S, MD_D, MD_R)^T @ (batch_size * MD_S, MD_D, MD_R)]
        # -> (batch_size * MD_S, N, MD_R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (batch_size * MD_S, MD_D, N) @ (batch_size * MD_S, N, MD_R) -> (batch_size * MD_S, MD_D, MD_R)
        numerator = torch.bmm(x, coef)
        # (batch_size * MD_S, MD_D, MD_R) @ [(batch_size * MD_S, N, MD_R)^T @ (batch_size * MD_S, N, MD_R)]
        # -> (batch_size * MD_S, D, MD_R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batch_size * MD_S, N, MD_R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (batch_size * MD_S, N, MD_R) @ (batch_size * MD_S, MD_D, MD_R)^T @ (batch_size * MD_S, MD_D, MD_R)
        # -> (batch_size * MD_S, N, MD_R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)
        return coef

"""
[batch_size, in_channels, height, width] -> [batch_size, out_channels, height // 4, width // 4]
"""
class StemConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            DownSampling(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
            DownSampling(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
        )

    def forward(self, x):
        out = self.proj(x)
        return out


class MSCA(nn.Module):

    def __init__(self, in_channels):
        super(MSCA, self).__init__()

        self.conv = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=(5, 5),
            padding=(2, 2),
            bias=True
        )

        self.conv7 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=True
            )
        )

        self.conv11 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                bias=True
            )
        )

        self.conv21 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                bias=True
            )
        )

        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        u = x
        out = self.conv(x)

        branch1 = self.conv7(out)
        branch2 = self.conv11(out)
        branch3 = self.conv21(out)

        out = self.fc(out + branch1 + branch2 + branch3)
        out = out * u
        return out


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )
        self.msca = MSCA(in_channels=in_channels)
        self.fc2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.msca(out)
        out = self.fc2(out)
        return out


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.):
        super(FFN, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=(1, 1)
        )
        self.dw = DepthwiseConv(
            in_channels=hidden_features,
            kernel_size=(3, 3),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=(1, 1)
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(self.dw(out))
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, in_channels, expand_ratio, drop_prob=0., drop_path_prob=0.):
        super(Block, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention(in_channels=in_channels)
        self.drop_path = DropPath(drop_prob=drop_path_prob if drop_path_prob >= 0 else nn.Identity)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(
            in_features=in_channels,
            hidden_features=int(expand_ratio * in_channels),
            out_features=in_channels,
            drop_prob=drop_prob
        )

        layer_scale_init_value = 1e-2
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.attention(out)
        out = x + self.drop_path(
            self.layer_scale1.unsqueeze(-1).unsqueeze(-1) * out
        )
        x = out

        out = self.norm2(out)
        out = self.ffn(out)
        out = x + self.drop_path(
            self.layer_scale2.unsqueeze(-1).unsqueeze(-1) * out
        )

        return out


class Stage(nn.Module):

    def __init__(
            self,
            stage_id,
            in_channels,
            out_channels,
            expand_ratio,
            blocks_num,
            drop_prob=0.,
            drop_path_prob=[0.]
    ):
        super(Stage, self).__init__()

        assert blocks_num == len(drop_path_prob)

        if stage_id == 0:
            self.down_sampling = StemConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.down_sampling = DownSampling(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2)
            )

        self.blocks = nn.Sequential(
            *[
                Block(
                    in_channels=out_channels,
                    expand_ratio=expand_ratio,
                    drop_prob=drop_prob,
                    drop_path_prob=drop_path_prob[i]
                ) for i in range(0, blocks_num)
            ]
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.down_sampling(x)
        out = self.blocks(out)
        # [batch_size, channels, height, width] -> [batch_size, channels, height * width]
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        # [batch_size, channels, height * width] -> [batch_size, height * width, channels]
        out = torch.transpose(out, -2, -1)
        out = self.norm(out)

        # [batch_size, height * width, channels] -> [batch_size, channels, height * width]
        out = torch.transpose(out, -2, -1)
        # [batch_size, channels, height * width] -> [batch_size, channels, height, width]
        out = out.view(batch_size, -1, height, width)

        return out


class MSCAN(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],
            expand_ratios=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob=0.1,
            drop_path_prob=0.1
    ):
        super(MSCAN, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]
        self.stages = nn.Sequential(
            *[
                Stage(
                    stage_id=stage_id,
                    in_channels=embed_dims[stage_id],
                    out_channels=embed_dims[stage_id + 1],
                    expand_ratio=expand_ratios[stage_id],
                    blocks_num=depths[stage_id],
                    drop_prob=drop_prob,
                    drop_path_prob=dpr[sum(depths[: stage_id]): sum(depths[: stage_id + 1])]
                ) for stage_id in range(0, len(depths))
            ]
        )

    def forward(self, x):
        out = x
        outputs = []

        for idx, stage in enumerate(self.stages):
            out = stage(out)
            outputs.append(out)

        # outputs: [output_of_stage1, output_of_stage2, output_of_stage3]
        # output_of_stage1: [batch_size, embed_dims[2], height / 8, width / 8]
        # output_of_stage2: [batch_size, embed_dims[3], height / 16, width / 16]
        # output_of_stage3: [batch_size, embed_dims[4], height / 32, width / 32]
        return [x, *outputs]


class Hamburger(nn.Module):

    def __init__(
            self,
            hamburger_channels=256,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(Hamburger, self).__init__()
        self.ham_in = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1)
            )
        )

        self.ham = NMF2D(args=nmf2d_config)

        self.ham_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hamburger_channels
            )
        )

    def forward(self, x):
        out = self.ham_in(x)
        out = self.ham(out)
        out = self.ham_out(out)
        out = F.relu(x + out)
        return out


class LightHamHead(nn.Module):

    def __init__(
            self,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            num_classes=150,
            drop_prob=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(LightHamHead, self).__init__()

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=num_classes,
                kernel_size=(1, 1)
            )
        )

        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels_list),
                out_channels=hidden_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),
            nn.ReLU()
        )

        self.hamburger = Hamburger(
            hamburger_channels=hidden_channels,
            nmf2d_config=nmf2d_config
        )

        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels
            ),
            nn.ReLU()
        )

    # inputs: [x, x_1, x_2, x_3]
    # x: [batch_size, channels, height, width]
    def forward(self, inputs):
        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)
        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        # x: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        x = torch.cat(inputs, dim=1)

        # out: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        out = self.squeeze(x)
        out = self.hamburger(out)
        out = self.align(out)

        # out: [batch_size, num_classes, standard_height, standard_width]
        out = self.cls_seg(out)

        _, _, original_height, original_width = o.shape
        # out: [batch_size, num_classes, original_height, original_width]
        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )
        # print('*********************')
        # print(out.view(batch_size, -1, original_height * original_width).shape)
        # out = torch.transpose(out.view(batch_size, -1, original_height * original_width), -2, -1)
        out = out.view(batch_size, -1, original_height, original_width)

        return out


# class SegNeXt(nn.Module):

#     def __init__(
#             self,
#             embed_dims=[3, 32, 64, 160, 256],
#             expand_rations=[8, 8, 4, 4],
#             depths=[3, 3, 5, 2],
#             drop_prob_of_encoder=0.1,
#             drop_path_prob=0.1,
#             hidden_channels=256,
#             out_channels=256,
#             num_classes=19,
#             drop_prob_of_decoder=0.1,
#             nmf2d_config=json.dumps(
#                 {
#                     "SPATIAL": True,
#                     "MD_S": 1,
#                     "MD_D": 512,
#                     "MD_R": 64,
#                     "TRAIN_STEPS": 6,
#                     "EVAL_STEPS": 7,
#                     "INV_T": 1,
#                     "ETA": 0.9,
#                     "RAND_INIT": True,
#                     "return_bases": False,
#                     "device": "cuda"
#                 }
#             )
#     ):
#         super(SegNeXt, self).__init__()

#         self.encoder = MSCAN(
#             embed_dims=embed_dims,
#             expand_ratios=expand_rations,
#             depths=depths,
#             drop_prob=drop_prob_of_encoder,
#             drop_path_prob=drop_path_prob
#         )

#         # self.decoder = LightHamHead(
#         #     in_channels_list=embed_dims[-3:],
#         #     hidden_channels=hidden_channels,
#         #     out_channels=out_channels,
#         #     num_classes=num_classes,
#         #     drop_prob=drop_prob_of_decoder,
#         #     nmf2d_config=nmf2d_config
#         # )

#     def forward(self, x):
#         out = self.encoder(x)
#         #out = self.decoder(out)
#         return out[1:]

class SegNeXt(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],
            expand_rations=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob_of_encoder=0.1,
            drop_path_prob=0.1,
        ):
        super(SegNeXt, self).__init__()

        self.encoder = MSCAN(
            embed_dims=embed_dims,
            expand_ratios=expand_rations,
            depths=depths,
            drop_prob=drop_prob_of_encoder,
            drop_path_prob=drop_path_prob
        )

    def forward(self, x):
        out = self.encoder(x)
        return out[1:]
    

@register_model
def SegNeXt_T(num_classes, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    embed_dims = [4, 96, 192, 384, 768]
    expand_rations = [8, 8, 4, 4]
    depths = [1, 1, 3, 1]

    net = SegNeXt(embed_dims=embed_dims, expand_rations=expand_rations,
                  depths=depths, **kwargs)
    return net

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")  
    
if __name__ == '__main__':
    from torchinfo import summary
    net = SegNeXt_T(num_classes=19)
    PrintModelInfo(net)
    summary(net, input_size=(1, 4, 320, 480))
    input = torch.randn(4, 320, 480)
    out=net(input)