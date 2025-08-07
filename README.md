# DiFusionSeg
This is official Pytorch implementation of "[DiFusionSeg: Diffusion-Driven Semantic Segmentation with Multi-modal Fusion for Perception Optimization]()"
 - 
```
@article{
}
```
## Framework
![image](./images/model_arch.png)

## Recommended Environment
 - [ ] torch  1.13.1
 - [ ] cudatoolkit 11.8
 - [ ] torchvision 0.14.0
 - [ ] mmcv  2.2.1
 - [ ] mmcv-full 1.7.2
 - [ ] mmsegmentation 0.30.0
 - [ ] numpy  1.26.4
 - [ ] opencv-python 4.10.0.84

## Experiments 
### Dataset & Checkpoints & Results
The checkpoints and results can be in [DiFusionSeg](https://www.dropbox.com/scl/fo/zjbyp7pml54epiz8wg4gj/AIGFfGfG8Ea_XU25WwyxQno?rlkey=1ywmahphox5f4kdqfrr8h1234&st=mag0vanh&dl=0). Download MSRS dataset from [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and the MFNet dataset from [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/).
If you need to evaluate other datasets, please organize them as follows:
```
├── /dataset
    MSRS/
    ├── test
    │   ├── ir
    │   ├── Segmentation_labels
    │   ├── Segmentation_visualize
    │   └── vi
    └── train
        ├── ir
        ├── Segmentation_labels
        └── vi
    MFD/
    ├── test
    │   ├── ir
    │   ├── Segmentation_labels
    │   ├── Segmentation_visualize
    │   └── vi
    ├── test_day
    │   ├── ir
    │   ├── Segmentation_labels
    │   └── vi
    ├── test_night
    │   ├── ir
    │   ├── Segmentation_labels
    │   └── vi
    └── train
        ├── ir
        ├── Segmentation_labels
        └── vi
    ......
```
### Evaluate model
python
```
python test_model.py
```
### run sample
python
```
python test_demo.py --img="./images/00131D_vi.png" --ir="./images/00131D_ir.png" --checkpoint="./exps/Done/msrs_vi_ir_meanstd_ConvNext_fusioncomplex_8083/best.pth" --segout="./seg.png"
```
### To Train
Before training DiFusionSeg, you need to download the MSRS dataset MSRS and putting it in ./datasets.

Then running 
python
```
python train_model.py
```
### Segmentation comparison
![image](./images/seg.png)
### Fusion comparison
![image](./images/fusion.png)
## If this work is helpful to you, please cite it as：
```
@article{
}
```
## Acknowledgements
