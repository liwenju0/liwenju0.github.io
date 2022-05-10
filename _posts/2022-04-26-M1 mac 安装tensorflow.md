---
layout: post
title:  "M1 mac 安装tensorflow"
date:   2022-04-26 09:20:08 +0800
category: "AI"
published: true
---

记录一下，后续可能还要使用。


<!--more-->

# 不能使用anaconda

如果安装了anaconda，必须先卸载。
先打开终端窗口并删除整个anaconda安装目录：rm -rf〜/ anaconda。

然后要编辑〜/ .bash_profile并从PATH环境变量中删除anaconda目录，并使用

rm -rf ~/.condarc 

~/.conda 

~/.continuum

删除可能在主目录中创建的.condarc文件和.conda以及.continuum目录

从这里下载miniforge。 https://github.com/conda-forge/miniforge/releases

 Mambaforge-4.12.0-0-MacOSX-arm64.sh

 安装，过程和Anaconda一样。

 # 依次执行如下命令

 conda install -c apple tensorflow-deps

 pip install tensorflow-metal

 pip install tensorflow-macos

 pip install tensorflow-datasets pandas jupyterlab


# 代码验证
```python
import tensorflow as tf
import tensorflow_datasets as tfds
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
```
