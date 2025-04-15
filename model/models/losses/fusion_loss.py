import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
# Parts of these codes are from: https://github.com/Linfeng-Tang/SeAFusion
"""SSIM"""
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret

class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=64, size_average=True, val_range=None):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
""""""
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()
        self.ssim_loss=SSIM_loss()
        self.alpha=1
        self.beta=0.8
        self.gamma=0.2

    def forward(self, image_vis, image_ir, generate_img):
        loss=dict()
        image_y = image_vis
        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = self.alpha*(F.l1_loss(generate_img, x_in_max))
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = self.beta*(F.l1_loss(generate_img_grad, x_grad_joint))
        loss_ssim=self.gamma*(self.ssim_loss(generate_img_grad.squeeze(2), image_vis) + 
                              self.ssim_loss(generate_img_grad.squeeze(2), image_ir))
        loss_fs = loss_in + loss_grad+loss_ssim
        loss["fusion_loss"]=loss_fs

        return loss