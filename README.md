# Earlier_Project

# 简介
本项目在王鹏宇学长的论文：《[A New Automatic Tool for CME Detection and Tracking with Machine Learning Techniques](https://arxiv.org/abs/1907.08798)》和代码的基础上，稍加改动并加以注释，力求接口简单易用。


# 项目目录
```
├── dataset.py
├── get_mask.py
├── model_result
│   └── LeNet5
├── models
│   ├── __init__.py
│   ├── lenet.py
│   ├── LeNet_PyTorch.py
│   ├── __pycache__
│   └── vgg16.py
├── __pycache__
│   └── dataset.cpython-36.pyc
├── README.md
├── rewrite
│   ├── co_loco_visual.py
│   ├── __init__.py
│   ├── __pycache__
│   └── refinement.py
├── runs
│   └── LeNet5
├── temporary_folder
│   ├── feature_maps
│   ├── refine_mask
│   ├── rough_mask
│   └── test_ori
├── test_LeNet.py
├── train_LeNet.py
├── update_label.json
└── wpy_cme
    ├── wpy_code_v2
    └── wpy_primary
```




`rewrite`文件夹下为重写的`Co-Localization`,`Refinement`代码，`wpy_cme`下为之前学长写的代码
# 运行
## 开发环境配置
所需基本环境：
+ Python 3.6+
+ PyTorch 1.0+
+ Numpy 1.19+
+ opencv-python=4.4.0.44
+ pymaxflow=1.2.13 [如何安装？](https://pypi.org/project/PyMaxflow/1.2.4/)
+ pillow=8.0.1
+ scikit-learn=0.23.2 ([如何安装？](https://scikit-learn.org/stable/install.html))
+ tqdm,argparse,torchsummary
> 部分库版本并不需要严格一致

## 直接测试
由于本项目已经有训练好的LeNet模型，可直接调用:
```python
python get_mask.py --testDir= <Your test image dir> --save_dir_mask= <the folder you specify to save the mask>
```
## 分段生成

### 1.如果想重新训练LeNet模型，可通过:
```python
python train_LeNet.py -h
```
来查看训练参数设置，并制定你的训练，模型结果文件夹将会有一个`json`文件来记录模型信息，包括最优模型路径

后续如果想换网络提取特征，也可自行撰写，只需要保证通过网络提取的feature map形状唯一并提前知道，后续代码中修改`FEATURE_MAP_SIZE`/`feature_map_size`即可。

**请注意**：`LeNet`中间层提取的`Feature Map`的大小为：`(25*25, 50)`,后续制定的`FEATURE_MAP_SIZE`必须与这里得到的`feature_map`的`size`相同。具体你可以自己指定。

### 2.提取feature maps
利用训练好的模型对测试图片进行测试，主要代码在`test_LeNet.py`文件里，提取的feature maps保存在`temporary_folder/feature_maps/`下

### 3.进行粗定位
代码在`rewrite/co_loco_visual.py`里，生成的rough mask保存在`temporary_folder/rough_mask`下，结果以`_roughmask.png`结尾

### 4.进行refine
代码在`rewrite/refinement.py`里，生成的refine mask保存在`temporary_folder/refine_mask`下，

其中：
+ 后缀为`_mask.png`是最终的单张mask图；
+ 后缀为`_mask_cat.png`是原图和mask组合图

待补充....2020.11.27
===
