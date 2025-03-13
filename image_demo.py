# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
os.chdir(sys.path[0])
from argparse import ArgumentParser
from model.apis import inference_segmentor, init_segmentor, show_result_pyplot,show_result_pyplot_self
from model.core.evaluation import get_palette
FILENAME="00038N.png"

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',default=f"./dataset/testimages/vi/{FILENAME}")
    parser.add_argument('--fusion_img', help='Image file',default=f"./out/fusion/{FILENAME}")
    parser.add_argument('--ir', help='ir file',default=f"./dataset/testimages/ir/{FILENAME}")
    parser.add_argument('--config', help='Config file',default="configs/ddp_config.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',default=
                        "./exps/Done/msrs_vi_ir_meanstd_ConvNext_fusioncomplex_8083/best.pth")
    parser.add_argument('--out-file', default=f"./out/seg/{FILENAME}", help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='msrs',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.3,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img,args.ir)
    # show the results
    # show_result_pyplot( 
    #     model,
    #     args.img,
    #     result,
    #     get_palette(args.palette),
    #     opacity=args.opacity,
    #     out_file=args.out_file)
    show_result_pyplot_self( 
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.out_file,
        fusion_img=args.fusion_img)


if __name__ == '__main__':
    main()
