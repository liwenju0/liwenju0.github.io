---
layout: post
title:  "使用pip完成tensorrt安装"
date:   2022-03-10 08:20:08 +0800
category: "AI"
published: true
---

> 使用pip安装tensorrt教程


<!--more-->

### 1、前提
使用pip安装完整可用的tensorrt有几个先决条件：
- python是3.6-3.9
- cuda是11.x
- linux操作系统，并且是x86-64的cpu架构。centos7以上，ubuntu18.04以上
- 若非root用户，使用pip时带上--user选项
  

### 2、安装步骤
python3 -m pip install --upgrade setuptools pip

python3 -m pip install nvidia-pyindex

python3 -m pip install --upgrade nvidia-tensorrt

上面的安装命令会拉取需要wheel形式的cuda和cudnn库，因为这些库是tensorrt wheel的依赖项。

如果安装时发生了如下错误：
##################################################################
The package you are trying to install is only a placeholder project on PyPI.org repository.
This package is hosted on NVIDIA Python Package Index.

This package can be installed as:
```
$ pip install nvidia-pyindex
$ pip install nvidia-tensorrt
```
##################################################################

首先检查python的版本是否是3.6-3.9。其次检查nvidia-pyindex是否安装成功，可以尝试卸载后重新安装。


### 3、检查安装是否成功
```python
import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())
```
上述代码如果没有报错，就代表安装成功。
可以尝试运行tensorrt提供的sample，应该都可以成功。




  



