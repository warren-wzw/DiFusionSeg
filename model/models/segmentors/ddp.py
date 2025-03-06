import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import cv2

from model.core import add_prefix
from model.ops import resize
from torch.special import expm1
from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
from sklearn.metrics import mutual_info_score

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from ..losses.fusion_loss import Fusionloss
from torchvision.transforms import ToPILImage

    
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

def save_single_image(img=None, ir=None, save_path_img=None, save_path_ir=None,size=None):
    if img is not None:
        file_name = save_path_img.split('/')[-1]
        file_name="./out/fusion/"+file_name
        img = resize(
            input=img,
            size=size,
            mode='bilinear',
            align_corners=False)
        img_np = img[0].permute(1, 2, 0).cpu().numpy()  # 变换维度
        img_np = (img_np * 255).astype(np.uint8)  # 反归一化到 [0, 255]
        cv2.imwrite(file_name, img_np)  # 保存可见光图像

    if ir is not None:
        file_name = save_path_img.split('/')[-1]
        file_name="./out/fusion/"+file_name
        ir = resize(
            input=ir,
            size=size,
            mode='bilinear',
            align_corners=False)
        ir_np = ir[0].squeeze(0).cpu().numpy()
        ir_np = (ir_np * 255).astype(np.uint8)  # 反归一化到 [0, 255]
        cv2.imwrite(save_path_ir, ir_np)  # 保存红外图像

def save_channels_as_images(tensor, output_dir='output', file_prefix='channel'):
    """
    将输入张量的每个通道保存为单独的图像文件。

    参数:
    - tensor: 输入的PyTorch张量，形状应为 [b, channels, height, width]。
    - output_dir: 输出图像文件的目录。
    - file_prefix: 输出图像文件的前缀。
    """
    # 确保输出目录存在
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将张量从 [b, channels, height, width] 转换为 [channels, b, height, width]
    tensor = tensor.permute(1, 0, 2, 3)

    # 初始化一个转换器，用于将张量转换为图像
    to_image = ToPILImage()

    # 遍历每个通道，并将每个通道保存为图像
    for i in range(tensor.shape[0]):  # 假设tensor.shape[0] 是通道数
        # 获取单个通道的张量
        single_channel = tensor[i]
        
        # 将张量转换为图像
        img = to_image(single_channel.squeeze())  # 去掉批次维度
        
        # 保存图像，文件名格式为 "<output_dir>/<file_prefix>_<channel_index>.png"
        img.save(os.path.join(output_dir, f'{file_prefix}_{i}.png'))
              
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class FusionModule_complex(nn.Module):
    def __init__(self):
        super(FusionModule_complex, self).__init__()

        # ----------------- 低分辨率分支处理 [b,256,80,120] -----------------
        # 第一层特征提取 (深度可分离卷积)
        self.low_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, groups=128),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 第二层特征提取
        self.low_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 并行路径
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128+64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # ----------------- 高分辨率分支处理 [b,4,320,480] -----------------
        self.high_conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # ----------------- 跨分辨率融合模块 -----------------
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(128+128, 256, 3, padding=1),  # 融合低分辨率上采样特征和高分辨率特征
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # ----------------- 输出模块 -----------------
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
        # ----------------- 注意力机制 -----------------
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        # ================= 低分辨率分支处理 =================
        # 第一阶段特征
        x_l1 = self.low_conv1(x_low)  # [b,128,80,120]
        # 并行路径
        x_dilated = self.dilated_conv(x_l1)  # [b,128,80,120]
        x_grouped = self.grouped_conv(self.low_conv2(x_l1))  # [b,64,80,120]
        # 特征融合
        low_fusion = torch.cat([x_dilated, x_grouped], dim=1)  # [b,192,80,120]
        low_fusion = self.fusion_conv(low_fusion)  # [b,128,80,120]
        # 上采样到高分辨率
        low_up = F.interpolate(low_fusion, scale_factor=4, mode='bilinear', align_corners=False)  # [b,128,320,480]
        #save_channels_as_images(low_up)
        # ================= 高分辨率分支处理 =================
        high_feat = self.high_conv(x_high)  # [b,128,320,480]
        #save_channels_as_images(high_feat)
        # ================= 跨分辨率融合 =================
        combined = torch.cat([low_up, high_feat], dim=1)  # [b,256,320,480]
        fused = self.cross_fusion(combined)  # [b,128,320,480]
        # ================= 最终输出 =================
        output = self.output_conv(fused)  # [b,3,320,480]
        #save_single_image(img=output,save_path_img="out.png",size=[480,640])
        # ================= 注意力增强 =================
        se_weight = self.se_block(output)  # [b,3,1,1]
        output = output * se_weight  # [b,3,320,480]

        return output

class FusionModule_simple(nn.Module):
    def __init__(self):
        super(FusionModule_simple, self).__init__()

        # ----------------- 低分辨率分支处理 -----------------

        # 第一层特征提取 (深度可分离卷积)
        self.low_conv1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, groups=64),  # 降低通道数
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 第二层特征提取
        self.low_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 并行路径
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.grouped_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=2),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64+32, 64, 1),  # 降低融合后的通道数
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ----------------- 高分辨率分支处理 -----------------
        self.high_conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ----------------- 跨分辨率融合模块 -----------------
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(64+64, 128, 3, padding=1),  # 降低卷积输出通道数
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ----------------- 输出模块 -----------------
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # ----------------- 注意力机制 -----------------
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 8, 1),  # 降低通道数
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        # ================= 低分辨率分支处理 =================
        x_l1 = self.low_conv1(x_low)  # [b,64,80,120]
        x_dilated = self.dilated_conv(x_l1)  # [b,64,80,120]
        x_grouped = self.grouped_conv(self.low_conv2(x_l1))  # [b,32,80,120]

        low_fusion = torch.cat([x_dilated, x_grouped], dim=1)  # [b,96,80,120]
        low_fusion = self.fusion_conv(low_fusion)  # [b,64,80,120]

        low_up = F.interpolate(low_fusion, scale_factor=4, mode='bilinear', align_corners=False)  # [b,64,320,480]

        # ================= 高分辨率分支处理 =================
        high_feat = self.high_conv(x_high)  # [b,64,320,480]

        # ================= 跨分辨率融合 =================
        combined = torch.cat([low_up, high_feat], dim=1)  # [b,128,320,480]
        fused = self.cross_fusion(combined)  # [b,64,320,480]

        # ================= 最终输出 =================
        output = self.output_conv(fused)  # [b,3,320,480]

        # ================= 注意力增强 =================
        se_weight = self.se_block(output)  # [b,3,1,1]
        output = output * se_weight  # [b,3,320,480]

        return output        

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 维持 256 维
        self.bn2 = nn.BatchNorm2d(256)
        
        self.seg_out = nn.Conv2d(256, num_classes, kernel_size=1)  # 输出语义分割
        self.feat_out = nn.Conv2d(256, 256, kernel_size=1)  # 生成 256 维特征

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        feat_fusion = F.relu(self.bn2(self.conv2(x)))  # 维持 256 维

        seg_out = self.seg_out(feat_fusion)  # 语义分割
        feat_out = self.feat_out(feat_fusion)  # 256 维特征

        # 下采样到 (B, 256, 80, 120)
        feat_out = F.avg_pool2d(feat_out, kernel_size=4, stride=4)  # 4x 下采样

        return seg_out, feat_out

@SEGMENTORS.register_module()
class DDP(EncoderDecoder):
    def __init__(self,
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion='ddim',
                 accumulation=False,
                 **kwargs):
        super(DDP, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.use_gt = False
        self.accumulation = accumulation
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.decode_head.in_channels[0])

        print(f" timesteps: {timesteps},"f" randsteps: {randsteps}, \
            "f" sample_range: {sample_range},"f" diffusion: {diffusion}")

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":#this
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.transform = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )
        self.fusion = FusionModule_complex()
        self.fusion_loss=Fusionloss()
        self.fusion_down= nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=3, stride=4, padding=1),  # 下采样并扩展通道
        nn.ReLU(inplace=True)
        )
        #self.fusionseg=SegmentationHead()
    """"""
    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times
    
    def encode_decode(self, img,ir, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        """"""
        img_ir = torch.cat([img, ir], dim=1)#[b,4,h,w]
        feature = self.extract_feat(img_ir)[0]#[b,256, h/4, w/4]
        """fusion"""
        fusion_out=self.fusion(feature,img_ir)
        """fusion with seg"""
        #_,feat_fusion=self.fusionseg(fusion_out)
        """fusion without seg"""
        feat_fusion=self.fusion_down(fusion_out)
        """"""
        feature_fusion = torch.cat([feature, feat_fusion], dim=1)
        feature_fusion = self.transform(feature_fusion)#turn b,512, h/4, w/4 to b,256, h/4, w/4
        """save out"""
        save_single_image(img=fusion_out,save_path_img=img_metas[0]['ori_filename'],
                          size=img_metas[0]['ori_shape'][:-1])
        """vi"""
        out = self.ddim_sample(feature_fusion,img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    @torch.no_grad()
    def ddim_sample(self, feature,img_metas):
        b, c, h, w, device = *feature.shape, feature.device #[b,256,h/4,w/4]
        time_pairs = self._get_sampling_timesteps(b, device=device)
        feature = repeat(feature, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([feature,mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr) #pad log_snr [1]-[1,1,1,1]
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next) #pad log_snr [1]-[1,1,1,1]
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)#[1,1024]
            mask_logit= self._decode_head_forward_test([feat], input_times)  # [bs, 256,h/4,w/4 ]-[b,9,h/4,w/4]
            mask_pred = torch.argmax(mask_logit, dim=1)
            """将预测的语义分割结果转化为预测的噪声"""
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale #scale to -0.01-0.01
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit
    
    def _decode_head_forward_test(self, x, t):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, t)
        return seg_logits

    def forward_train(self, img, img_metas, ir,img_ori,ir_ori,gt_semantic_seg):
        losses = dict()
        """create input"""
        # save_single_image(img=img_ori,ir=ir_ori)
        img_ir = torch.cat([img, ir], dim=1)
        """image"""
        feature = self.extract_feat(img_ir)[0]  # bs, 256, h/4, w/4
        """fusion"""
        img_ori,ir_ori=img_ori.float(),ir_ori.float()
        fusion_out=self.fusion(feature,img_ir)#[b,3,h,w]
        loss_fusion=self.fusion_loss(img_ori,ir_ori,fusion_out)
        """fusion with seg"""
        # fusion_seg,feat_fusion=self.fusionseg(fusion_out)
        # loss_seg = F.cross_entropy(fusion_seg, gt_semantic_seg.squeeze(1),ignore_index=255)
        # loss_fusion["seg_loss"]=loss_seg
        """fusion without seg"""
        feat_fusion=self.fusion_down(fusion_out)#b,256,h/4, w/4
        """"""
        feature_fusion = torch.cat([feature, feat_fusion], dim=1)
        feature_fusion = self.transform(feature_fusion)#turn b,512,h/4, w/4 to b,256,h/4, w/4
        losses.update(loss_fusion)
        """gtdown represents the embedding of semantic segmentation labels after downsampling"""
        batch, c, h, w, device, = *feature.shape, feature.device
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes
        gt_down = self.embedding_table(gt_down).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale
        """sample time"""
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],self.sample_range[1])  # [bs]
        """random noise"""
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img_ir, noise_level)#turn [b]->[b,1,1,1]
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise
        """cat input and noise"""
        feat = torch.cat([feature_fusion, noised_gt], dim=1)
        feat = self.transform(feat)#turn b,512,h/4, w/4 to b,256,h/4, w/4
        """conditional input"""
        input_times = self.time_mlp(noise_level)
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg,img_ori,ir_ori)
        losses.update(loss_decode)
        """aux seg head"""
        loss_aux = self._auxiliary_head_forward_train([feature], img_metas, gt_semantic_seg)
        losses.update(loss_aux)
        return losses

    def _decode_head_forward_train(self, x, input_times, img_metas, gt_semantic_seg,img_ori,ir_ori):#feature time image infos,groundtruth
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, 
                                                     input_times, 
                                                     img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     img_ori,ir_ori)

        return loss_decode

    
