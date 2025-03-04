# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import re
os.chdir(sys.path[0])
from argparse import ArgumentParser

from model.apis import inference_segmentor, init_segmentor, show_result_pyplot
from model.core.evaluation import get_palette
FILENAME="./dataset/PST/test/"

# 自然排序函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main():
    parser = ArgumentParser()
    parser.add_argument('--img_file', help='Image file',default=f"{FILENAME}vi")
    parser.add_argument('--ir_file', help='ir file',default=f"{FILENAME}ir")
    parser.add_argument('--config', help='Config file',default="configs/ddp_config.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',default=
                        "./exps/Done/msrs_vi_ir_meanstd_ConvNext_fusioncomplex_8083/48000_14400.pth")
    parser.add_argument('--out-file', default=f"./out/vi_ir/{FILENAME}", help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='msrs',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    img_list = sorted(os.listdir(args.img_file), key=natural_sort_key)
    ir_list = sorted(os.listdir(args.ir_file), key=natural_sort_key)
    for img,ir in zip(img_list,ir_list):
        if img==ir:
            img=FILENAME+"vi/"+img
            ir=FILENAME+"ir/"+ir
            result = inference_segmentor(model, img,ir)
            print(f"{img}")
        else:
            return 

if __name__ == '__main__':
    main()