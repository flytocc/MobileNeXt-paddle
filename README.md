# Rethinking Bottleneck Structure for Efficient Mobile Network Design

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
   * [2.1 数据集](#21-数据集)
   * [2.2 复现精度](#22-复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. 代码结构](#5-代码结构)
* [6. 自动化测试脚本](#6-自动化测试脚本)
* [7. License](#7-license)
* [8. 参考链接与文献](#8-参考链接与文献)


## 1. 简介

这是一个PaddlePaddle实现的 MobileNeXt 。

**论文:**
[Rethinking Bottleneck Structure for Efficient Mobile Network Design](https://arxiv.org/pdf/2007.02269.pdf)

**参考repo:**
[MobileNeXt](https://github.com/yitu-opensource/MobileNeXt) & 
[rethinking_bottleneck_design](https://github.com/zhoudaquan/rethinking_bottleneck_design)

在此非常感谢`zhoudaquan`和`yitutech-opensource`等人的贡献，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

### 2.1 数据集

[ImageNet](https://image-net.org/)项目是一个大型视觉数据库，用于视觉目标识别研究任务，该项目已手动标注了 1400 多万张图像。ImageNet-1k 是 ImageNet 数据集的子集，其包含 1000 个类别。训练集包含 1281167 个图像数据，验证集包含 50000 个图像数据。2010 年以来，ImageNet 项目每年举办一次图像分类竞赛，即 ImageNet 大规模视觉识别挑战赛（ILSVRC）。挑战赛使用的数据集即为 ImageNet-1k。到目前为止，ImageNet-1k 已经成为计算机视觉领域发展的最重要的数据集之一，其促进了整个计算机视觉的发展，很多计算机视觉下游任务的初始化模型都是基于该数据集训练得到的。

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

### 2.2 复现精度

| 模型            | epochs | top1 acc (参考精度) | (复现精度) | 权重                \| 训练日志 |
|:--------------:|:------:|:------------------:|:---------:|:-----------------------------:|
| MobileNeXt-1.0 |  200   | 74.022             | 74.022    | checkpoint-best.pd \| log.txt |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1Kt5Bk6PhlrCSs4Ie5hwamg?pwd=cp32)


## 3. 准备数据与环境

### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * 3090
- 框架：
  - PaddlePaddle == 2.3.1
  - Pillow == 8.4.0

* 安装paddlepaddle

```bash
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.3.1
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/MobileNeXt-paddle.git
cd MobileNeXt-paddle
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

参考 [2.1 数据集](#21-数据集)，从官方下载数据后，按如下格式组织数据，即可在 PaddleClas 中使用 ImageNet1k 数据集进行训练。

```bash
imagenet/
    |_ train/
    |  |_ n01440764
    |  |  |_ n01440764_10026.JPEG
    |  |  |_ ...
    |  |_ ...
    |  |
    |  |_ n15075141
    |     |_ ...
    |     |_ n15075141_9993.JPEG
    |_ val/
    |  |_ ILSVRC2012_val_00000001.JPEG
    |  |_ ...
    |  |_ ILSVRC2012_val_00050000.JPEG
    |_ train_list.txt
    |_ val_list.txt
```


## 4. 开始使用

### 4.1 模型训练

* 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus=0,1,2,3 \
    main.py \
    /path/to/imagenet/ \
    # --cls_label_path_train /path/to/train_list.txt \
    # --cls_label_path_val /path/to/val_list.txt \
    --model MobileNeXt-1.0 \
    --interpolation random \
    --batch_size 128 \
    --use_nesterov --weight_decay 1e-4 --t_in_epochs \
    --lr 0.1 --min_lr 1e-5 --epochs 200 \
    --color_jitter 0.4 \
    --model_ema \
    --num_workers 10 \
    --use_multi_epochs_loader \
    --output ./output/
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_dir`下train/val里的图片作为train-set/val-set。

部分训练日志如下所示。

```
Epoch: [15]  [1900/2502]  eta: 0:02:56  lr: 0.099087  loss: 3.1850 (3.1407)  time: 0.2930  data: 0.0017
Epoch: [15]  [1950/2502]  eta: 0:02:41  lr: 0.099087  loss: 3.1298 (3.1401)  time: 0.2977  data: 0.0017
```

### 4.2 模型评估

``` shell
python eval.py \
    /path/to/imagenet/ \
    # --cls_label_path_val /path/to/val_list.txt \
    --model MobileNeXt-1.0 \
    --batch_size 256 \
    --interpolation bicubic \
    --resume $TRAINED_MODEL
```

ps: 如果未指定`cls_label_path_val`，会读取`data_dir`/val里的图片作为val-set。

### 4.3 模型预测

```shell
python predict.py \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --model MobileNeXt-1.0 \
    --interpolation bicubic \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 246, 236, 209], 'scores': [0.8502025604248047, 0.01571478880941868, 0.013417884707450867, 0.003637149930000305, 0.003541758982464671], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'Chesapeake Bay retriever']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8502025604248047`。

### 4.4 模型导出

```shell
python export_model.py \
    --model MobileNeXt-1.0 \
    --output /path/to/save/export_model/ \
    --resume $TRAINED_MODEL

python infer.py \
    --interpolation bicubic \
    --model_file /path/to/save/export_model/output/model.pdmodel \
    --params_file /path/to/save/export_model/output/model.pdiparams \
    --input_file ./demo/ILSVRC2012_val_00020010.JPEG
```

输出结果为
```
[{'class_ids': [178, 211, 246, 236, 209], 'scores': [0.8507419228553772, 0.015670042484998703, 0.013438238762319088, 0.0036271170247346163, 0.0035393161233514547], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'Chesapeake Bay retriever']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8507419228553772`。与predict.py结果的误差在正常范围内。


## 5. 代码结构

```
├── data
├── demo
├── test_tipc
├── util
├── engine.py
├── eval.py
├── export_model.py
├── infer.py
├── main.py
├── models.py
├── predict.py
├── README.md
└── requirements.txt
```


## 6. 自动化测试脚本

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-*-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/config/MobileNeXt/MobileNeXt-1.0.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh test_tipc/config/MobileNeXt/MobileNeXt-1.0.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
Run successfully with command - python3 main.py ./dataset/ILSVRC2012/ --cls_label_path_train=./dataset/ILSVRC2012/train_list.txt --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --model=MobileNeXt-1.0 --interpolation=bicubic --use_nesterov --weight_decay=1e-4 --lr=0.1 --min_lr=1e-5 --color_jitter=0.4 --model_ema --use_multi_epochs_loader --experiment .    --output=./test_tipc/output/norm_train_gpus_0_autocast_null/MobileNeXt-1.0 --epochs=2     --batch_size=8   !
Run successfully with command - python3 eval.py ./dataset/ILSVRC2012/ --cls_label_path_val=./dataset/ILSVRC2012/val_list.txt --model=MobileNeXt-1.0 --interpolation=bicubic --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/MobileNeXt-1.0/checkpoint-latest.pd    !
Run successfully with command - python3 export_model.py --model=MobileNeXt-1.0 --resume=./test_tipc/output/norm_train_gpus_0_autocast_null/MobileNeXt-1.0/checkpoint-latest.pd --output=./test_tipc/output/norm_train_gpus_0_autocast_null!
Run successfully with command - python3 infer.py --use_gpu=True --use_tensorrt=False --precision=fp32 --model_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdmodel --batch_size=1 --input_file=./dataset/ILSVRC2012/val  --params_file=./test_tipc/output/norm_train_gpus_0_autocast_null/model.pdiparams > ./test_tipc/output/python_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log 2>&1 !
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。


## 7. License

This project is released under BSD License.


## 8. 参考链接与文献

1. Rethinking Bottleneck Structure for Efficient Mobile Network Design: https://arxiv.org/pdf/2007.02269.pdf
2. MobileNeXt: https://github.com/yitu-opensource/MobileNeXt
3. rethinking_bottleneck_design: https://github.com/zhoudaquan/rethinking_bottleneck_design

```
@article{zhou2020rethinking,
  title={Rethinking Bottleneck Structure for Efficient Mobile Network Design},
  author={Zhou, Daquan and Hou, Qibin and Chen, Yunpeng and Feng, Jiashi and Yan, Shuicheng},
  journal={ECCV, August},
  year={2020}
}
```