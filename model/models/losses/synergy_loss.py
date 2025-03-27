import torch
import torch.nn as nn
import torch.nn.functional as F


class SynergyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SynergyLoss, self).__init__()
        self.eps = eps

    def forward(self, fused_img, ir_img, rgb_img, seg_pred):
        """
        Calculate synergyLoss
        """
        loss=dict()
        # get seg
        seg_mask = torch.argmax(seg_pred, dim=1, keepdim=True).float() 

        # calculate mse loss in seg part
        L_synergy_ir = F.mse_loss(fused_img * seg_mask, ir_img * seg_mask)  
        L_synergy_rgb = F.mse_loss(fused_img * seg_mask, rgb_img * seg_mask)

        # total
        L_synergy = L_synergy_ir + L_synergy_rgb
        loss["sync_loss"]=L_synergy

        return loss
