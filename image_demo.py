# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
os.chdir(sys.path[0])
from argparse import ArgumentParser
from model.apis import inference_segmentor, init_segmentor, show_result_pyplot
from model.core.evaluation import get_palette
FILENAME="2.png"

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',default=f"./dataset/testimages/vi/{FILENAME}")
    parser.add_argument('--ir', help='ir file',default=f"./dataset/testimages/ir/{FILENAME}")
    parser.add_argument('--config', help='Config file',default="configs/ddp_config.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',default=
                        "./exps/Done/msrs_vi_ir_meanstd_ConvNext_fusioncomplex_8083/best_mIoU_iter_48000.pth")
    parser.add_argument('--out-file', default=f"./out/seg/{FILENAME}", help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='msrs',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1.0,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img,args.ir)
    # show the results
    show_result_pyplot( 
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
