from asyncio.unix_events import BaseChildWatcher
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
import torch
import warnings
from model.models.builder import HEADS
from model.models.decode_heads.decode_head import BaseDecodeHead
from model.ops import resize
from torchvision.transforms import ToPILImage
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)
from torch.nn.init import normal_

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
import numpy as np
from PIL import Image 
 
def get_palette():
    unlabeled = [0,0,0]
    car        = [255, 82, 82]
    person     = [179, 57, 57]
    bike       = [204, 142, 53]
    curve      = [205, 97, 51]
    car_stop   = [51, 217, 178]
    guardrail  = [255, 177, 66]
    color_cone = [112, 111, 211]
    bump       = [52, 172, 224]
    palette    = np.array([unlabeled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette
        
def visualize(image_name, predictions):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)):
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(f'output/'  + image_name + '.png')
        
@HEADS.register_module()
class DeformableHeadWithTime(BaseDecodeHead):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """
    
    def __init__(self,
                 num_feature_levels,
                 encoder,
                 positional_encoding,
                 **kwargs):
        
        super().__init__(input_transform='multiple_select', **kwargs)
    
        self.num_feature_levels = num_feature_levels
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        # normal_(self.level_embeds)
    
    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    
    @auto_fp16()
    def forward(self, inputs, times):#input [b,256,80,120],time:[8,1024]
        mlvl_feats = inputs[-self.num_feature_levels:]#mlvl_feats [b,256,80,120]
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):#only one
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)#[b,h*w,256]
            feat = feat.flatten(2).transpose(1, 2)#[b,h*w,256]
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)#[b,h*w,256]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)#[b,h*w,256]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(  #DetrTransformerEncoder
            query=feat_flatten,
            key=None,
            value=None,
            time=times,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index)# (H*W, bs, embed_dims)
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        # single_channel_image = memory.mean(dim=1).unsqueeze(0)
        # save_channels_as_images(single_channel_image)
        seg_out = self.conv_seg(memory)#[batch,256,80,120]->[batch,9,80,120]
        #visualize("seg",seg_out.argmax(1))
        return seg_out
    
    def forward_train(self, inputs, times, img_metas, gt_semantic_seg, train_cfg,img_ori,ir_ori):
        seg_logits = self.forward(inputs, times)#[b,numclssses,80,120]
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, times):
        out=self.forward(inputs, times)
        return out
       
    def forward_train_return_logits(self, inputs, times, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, times)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, seg_logits
