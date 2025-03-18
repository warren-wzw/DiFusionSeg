import torch
import torch.nn as nn
import torch.nn.functional as F


class SynergyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SynergyLoss, self).__init__()
        self.eps = eps

    def forward(self, fused_img, ir_img, rgb_img, seg_pred):
        """
        计算协同约束损失，其中只计算分割区域内的误差
        """
        loss=dict()
        # 获取分割掩码
        seg_mask = torch.argmax(seg_pred, dim=1, keepdim=True).float()  # 分割掩码，取每个像素的最大类别

        # 计算在分割区域内的 MSE 损失
        L_synergy_ir = F.mse_loss(fused_img * seg_mask, ir_img * seg_mask)  # 红外图像与融合图像
        L_synergy_rgb = F.mse_loss(fused_img * seg_mask, rgb_img * seg_mask)  # 可见光图像与融合图像

        # 总协同约束损失
        L_synergy = L_synergy_ir + L_synergy_rgb
        loss["sync_loss"]=L_synergy

        return loss
