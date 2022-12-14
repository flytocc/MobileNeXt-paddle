
# 飞桨训推一体认证

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleClas中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/guide.png" width="1000">
</div>

## 2. 汇总信息

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 |
| :------ | :--------- | :---- | :------- | :---------------- |
| MobileNeXt | MobileNeXt-1.0  | 分类  | 支持     | 多机多卡 <br> 混合精度 |

## 3. 测试工具简介
### 目录介绍
```
./test_tipc/
├── config                              # 配置文件目录
│   └── MobileNeXt                        # MobileNeXt 系列模型测试配置文件目录
│       └── MobileNeXt-1.0.txt                # 基础训练预测配置文件
├── docs
│   ├── guide.png
│   ├── install.md                      # 配置运行环境
│   ├── test_train_inference_python.md  # Linux端基础训练预测功能测试文档
├── common_func.sh                      # test_*.sh会调用到的公共函数
├── prepare.sh                          # 完成test_*.sh运行所需要的数据和模型下载
├── README.md                           # 使用文档
└── test_train_inference_python.sh      # 测试python训练预测的主程序
```

### 测试流程概述
使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程如下：

1. 运行`prepare.sh`准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_*.sh`，产出log，由log可以看到不同配置是否运行成功；

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：

```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh  configs/[model_name]/[params_file_name]  [Mode]

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]

```

例如，测试基本训练预测功能的`lite_train_lite_infer`模式，运行：

```shell
# 准备数据
bash test_tipc/prepare.sh test_tipc/config/MobileNeXt/MobileNeXt-1.0.txt 'lite_train_lite_infer'

# 运行测试
bash test_tipc/test_train_inference_python.sh test_tipc/config/MobileNeXt/MobileNeXt-1.0.txt 'lite_train_lite_infer'
```

关于本示例命令的更多信息可查看[基础训练预测使用文档](docs/test_train_inference_python.md)。

### 配置文件命名规范

在`configs`目录下，**按模型系列划分为子目录**，子目录中存放所有该模型系列测试需要用到的配置文件，如`MobileNeXt`文件夹下存放了所有`MobileNeXt`系列模型的配置文件。配置文件的命名遵循如下规范：

1. 基础训练预测配置简单命名为：`ModelName_train_infer_python.txt`，表示**Linux环境下单机、不使用混合精度训练+python预测**，其完整命名对应`ModelName_train_linux_gpu_normal_normal_infer_python_linux_gpu_cpu.txt`，由于本配置文件使用频率较高，这里进行了名称简化。其中`ModelName`指具体模型名称
2. 其他带训练配置命名格式为：`ModelName_train_训练硬件环境(linux_gpu/linux_dcu/…)_是否多机(fleet/normal)_是否混合精度(amp/normal)_预测模式(infer/lite/serving/js)_语言(cpp/python/java)_预测硬件环境(ModelName_linux_gpu/mac/jetson/opencl_arm_gpu/...).txt`。如，linux gpu下多机多卡+混合精度链条测试对应配置 `ModelName_train_linux_gpu_fleet_amp_infer_python_linux_gpu_cpu.txt`，linux dcu下基础训练预测对应配置 `ModelName_train_linux_dcu_normal_normal_infer_python_linux_dcu.txt`。
3. 仅预测的配置（如serving、lite等）命名格式：`ModelName_model_训练硬件环境(ModelName_linux_gpu/linux_dcu/…)_是否多机(fleet/normal)_是否混合精度(amp/normal)_(infer/lite/serving/js)_语言(cpp/python/java)_预测硬件环境(ModelName_linux_gpu/mac/jetson/opencl_arm_gpu/...).txt`，即，与2相比，仅第二个字段从train换为model，测试时模型直接下载获取，这里的“训练硬件环境”表示所测试的模型是在哪种环境下训练得到的。

**根据上述命名规范，可以直接从子目录名称和配置文件名找到需要测试的场景和功能对应的配置文件。**

## 4 开始测试

各功能测试中涉及混合精度等训练相关，及mkldnn、Tensorrt等多种预测相关参数配置，请点击下方相应链接了解更多细节和使用教程：  

- [test_train_inference_python 使用](docs/test_train_inference_python.md)：测试基于Python的模型训练、评估、推理等基本功能。
