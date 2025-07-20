import torch
import torch.nn.functional as F
import cv2
import os
from matplotlib import cm
import matplotlib.pyplot as plt

def visualize_feature_activations(feature, original_img, ir_img, img_metas, save_dir='./out/heatmap/heatmaps'):
    """可视化特征激活热力图"""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from torch.nn import functional as F
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 取特征的平均激活值作为注意力图
    # feature形状: [b, 256, h/4, w/4]
    feature_map = feature[0].mean(dim=0)  # [h/4, w/4]
    
    # 上采样到原始图像尺寸
    feature_map = F.interpolate(
        feature_map.unsqueeze(0).unsqueeze(0),  # [1, 1, h/4, w/4]
        size=original_img.shape[2:],  # 原始图像尺寸
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # 归一化到0-1
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    
    # 将原始RGB图像转为numpy数组并归一化
    rgb_img = original_img[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    # 绘制可视化图像
    plt.figure(figsize=(15, 5))
    
    # 原始RGB图像
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(feature_map, cmap='jet')
    plt.title('Attention Heatmap')
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_img)
    plt.imshow(feature_map, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    # 保存图像
    filename = os.path.basename(img_metas[0]['filename']).split('.')[0]
    plt.savefig(f"{save_dir}/{filename}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {save_dir}/{filename}_heatmap.png")
   
def  visualize_prediction_heatmap(out, img, img_metas, save_dir='./out/heatmap/pred_heatmaps'):
    """可视化预测结果热力图"""
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(img_metas[0]['filename']).split('.')[0]
    
    # 获取每个类别的最大概率值
    out_softmax = F.softmax(out, dim=1)
    max_prob, _ = torch.max(out_softmax, dim=1)
    confidence_map = max_prob[0].cpu().numpy()
    
    # 获取原始图像
    rgb_img = img[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # 置信度热力图
    plt.subplot(1, 3, 2)
    plt.imshow(confidence_map, cmap='jet')
    plt.title('Confidence Heatmap')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_img)
    plt.imshow(confidence_map, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.savefig(f"{save_dir}/{filename}_pred_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction heatmap saved to {save_dir}/{filename}_pred_heatmap.png")

def visualize_fusion_features(pre_fusion, post_fusion, img, img_metas, save_dir='./out/heatmap/fusion_vis'):
    """可视化融合前后的特征变化"""
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.nn import functional as F
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(img_metas[0]['filename']).split('.')[0]
    
    # 获取激活图
    pre_map = pre_fusion[0].mean(dim=0)
    post_map = post_fusion[0].mean(dim=0)
    
    # 上采样到原始图像尺寸
    pre_map = F.interpolate(
        pre_map.unsqueeze(0).unsqueeze(0),
        size=img.shape[2:],
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    post_map = F.interpolate(
        post_map.unsqueeze(0).unsqueeze(0),
        size=img.shape[2:],
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # 归一化
    pre_map = (pre_map - pre_map.min()) / (pre_map.max() - pre_map.min() + 1e-8)
    post_map = (post_map - post_map.min()) / (post_map.max() - post_map.min() + 1e-8)
    
    # 计算差异图
    diff_map = np.abs(post_map - pre_map)
    diff_map = diff_map / diff_map.max()
    
    # 获取原始图像
    rgb_img = img[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    """"""
    pre_map_uint8 = np.uint8((pre_map) * 255)
    pre_map_color = cv2.applyColorMap(pre_map_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{filename}_Pre-fusion.png", pre_map_color)
    post_map_uint8 = np.uint8((post_map) * 255)
    post_map_color = cv2.applyColorMap(post_map_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{filename}_Post-fusion.png", post_map_color)
    colormap = cm.get_cmap('coolwarm')
    colored_img = colormap(diff_map)[:, :, :3]  # 只取 RGB，去掉 alpha 通道
    # 4. 转为 0~255 的 uint8 格式，并从 RGB 转为 BGR（OpenCV 显示）
    colored_img_uint8 = (colored_img * 255).astype(np.uint8)
    colored_img_bgr = cv2.cvtColor(colored_img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_dir}/{filename}_Diff.png", colored_img_bgr)
           
    # 可视化
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # 融合前热力图
    plt.subplot(2, 3, 2)
    plt.imshow(pre_map, cmap='jet')
    plt.title('Pre-fusion Attention')
    plt.axis('off')
    
    # 融合前叠加图
    plt.subplot(2, 3, 3)
    plt.imshow(rgb_img)
    plt.imshow(pre_map, cmap='jet', alpha=0.5)
    plt.title('Pre-fusion Overlay')
    plt.axis('off')
    
    # 融合后热力图
    plt.subplot(2, 3, 5)
    plt.imshow(post_map, cmap='jet')
    plt.title('Post-fusion Attention')
    plt.axis('off')
    
    # 融合后叠加图
    plt.subplot(2, 3, 6)
    plt.imshow(rgb_img)
    plt.imshow(post_map, cmap='jet', alpha=0.5)
    plt.title('Post-fusion Overlay')
    plt.axis('off')
    
    # 差异图
    plt.subplot(2, 3, 4)
    plt.imshow(diff_map, cmap='coolwarm')
    plt.title('Attention Difference')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}_fusion_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fusion comparison saved to {save_dir}/{filename}_fusion_comparison.png")

def save_heatmap(feature_map, save_path="heatmaps/low_fusion.png"):
    """
    Save a feature map as a clean heatmap image without titles, axes, or colorbars.

    Args:
        feature_map (Tensor): Shape [B, C, H, W].
        save_path (str): Path to save the heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fmap = feature_map[0]  # 取第一个 batch
    heatmap = torch.mean(fmap, dim=0).detach().cpu()  # [H, W]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)  # 归一化

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')     # 关闭坐标轴
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Heatmap saved to {save_path}")