# Copyright (c) OpenMMLab. All rights reserved.
# from .unused.ann_head import ANNHead
# from .unused.apc_head import APCHead
# from .unused.aspp_head import ASPPHead
# from .unused.cc_head import CCHead
# from .unused.da_head import DAHead
# from .unused.dm_head import DMHead
# from .unused.dnl_head import DNLHead
# from .unused.dpt_head import DPTHead
# from .unused.ema_head import EMAHead
# from .unused.enc_head import EncHead
# from .unused.fpn_head import FPNHead
# from .unused.gc_head import GCHead
# from .unused.isa_head import ISAHead
# from .unused.knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
# from .unused.lraspp_head import LRASPPHead
# from .unused.nl_head import NLHead
# from .unused.ocr_head import OCRHead
# from .unused.point_head import PointHead
# from .unused.psa_head import PSAHead
# from .unused.psp_head import PSPHead
# from .unused.segformer_head import SegformerHead
# from .unused.sep_aspp_head import DepthwiseSeparableASPPHead
# from .unused.sep_fcn_head import DepthwiseSeparableFCNHead
# from .unused.setr_mla_head import SETRMLAHead
# from .unused.setr_up_head import SETRUPHead
# from .unused.stdc_head import STDCHead
# from .unused.uper_head import UPerHead
# from .unused.nn_head import NNHead
# from .unused.fcn_head_with_time import FCNHeadWithTime
# from .unused.identity_head import IdentityHead

#from .deformable_head import DeformableHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .deformable_head_with_time import DeformableHeadWithTime
from .fcn_head import FCNHead


__all__ = ['SegmenterMaskTransformerHead','DeformableHeadWithTime','FCNHead']

# __all__ = [
#     'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
#     'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
#     'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
#     'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
#     'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
#     'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
#     'KernelUpdateHead', 'KernelUpdator', 'NNHead', 'DeformableHead',
#     'DeformableHeadWithTime', 'FCNHeadWithTime', 'IdentityHead'
# ]
