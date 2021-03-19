## Description
This is a PyTorch Implementation of [DCLNet](https://openaccess.thecvf.com/content/WACV2021/papers/Bi_Disentangled_Contour_Learning_for_Quadrilateral_Text_Detection_WACV_2021_paper.pdf) &quot; Disentangled Contour Learning for Quadrilateral Text Detection &quot;.

| ICDAR2017 MLT | Precision | Recall | F-score | 
| - | - | - | - |
| DCLNet | 81.0 | 66.9 | 73.3 |
| DCLNet* | 81.9 | 71.4 | 76.3 |

## Prerequisites
Only tested on
* Anaconda3
* python 3.7.1
* torch 1.2.0
* torchvision 0.4.0
* opencv-python 4.4.0.42
* easydict 1.9

## Installation
### 1. LANMS

Refer to [LANMS](https://github.com/SakuraRiven/LANMS)

### 2. DCN

Refer to DCN in [DBNet](https://github.com/MhLiao/DB). Note that this repo puts `dcn` in the `model` dir.

### 3. Clone
```
git clone https://github.com/SakuraRiven/DCLNet.git
cd DCLNet
```

### 4. Data & Pre-Trained Model
* [ICDAR2017](https://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1), [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)

* [backbone](https://drive.google.com/file/d/1gCsHA-ZTx-_UT6JJGVRugj0QOwonmUaP/view?usp=sharing)

* [pretrained model](https://drive.google.com/file/d/1bjNQCli10Zlhr6IfgUhIRHYyfmxY0r8x/view?usp=sharing)

* [finetune model](https://drive.google.com/file/d/1w0g3ewNW0gkC2iYcj1PdJoxeYXEU0F1N/view?usp=sharing)

Make a new folder ```pths``` and put the download pths into ```pths```

```
mkdir pths
cd pths
mkdir backbone pretrain train
cd ..
mv resnet50-19c8e357.pth pths/backbone/
mv model_epoch_5.pth pths/pretrain/
mv model_epoch_150.pth pths/train/
```

Here is an example:
```
.
├── DCLNet
│   ├── model
│   │   └── dcn
│   └── pths
│       ├── backbone
│       ├── pretrain
│       └── train
└── data
    ├── ICDAR2017
    │   ├── train_img
    │   ├── train_gt
    │   ├── valid_img
    │   ├── valid_gt
    │   └── test_img
    └── SynthText
```

## Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py pretrain
```
## Finetune
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py finetune
```
## Evaluate
```
CUDA_VISIBLE_DEVICES=0 python eval.py
CUDA_VISIBLE_DEVICES=0 python multi_scale_eval.py
```
## Detect
```
CUDA_VISIBLE_DEVICES=0 python detect.py
```
## Citation
Please cite the related works in your publications if it helps your research:
```
@inproceedings{bi2021disentangled,
  title={Disentangled Contour Learning for Quadrilateral Text Detection},
  author={Bi, Yanguang and Hu, Zhiqiang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={909--918},
  year={2021}
}
```
