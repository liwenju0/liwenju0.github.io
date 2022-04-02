---
layout: post
title:  "onnxruntime使用gpu推理"
date:   2022-04-02 09:20:08 +0800
category: "AI"
published: true
---
之前踩过的一个坑，有小伙伴问，索性记录下来，免得忘记。


# 1、gpu版本的onnxruntime

首先要强调的是，有两个版本的onnxruntime，一个叫onnxruntime，只能使用cpu推理，另一个叫onnxruntime-gpu，既可以使用gpu，也可以使用cpu。

如果自己安装的是onnxruntime，需要卸载后安装gpu版本。

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.9
```
<!--more-->

# 2、 确认一下是否可以使用gpu
注意：
```python
print(onnxruntime.get_device())
```
上面的代码给出的输出是'GPU'时，并不代表就成功了。

而要用下面的代码来验证：

```python
ort_session = onnxruntime.InferenceSession("path/model/model_name.onnx",
                                               providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())
```

当输出是:['CUDAExecutionProvider', 'CPUExecutionProvider']才表示成功了。

# 3、配置cuda
如果上面的输出不对，可能需要配置下cuda。

进入/usr/local目录下，查看是否有cuda。

![2022-04-02-onnxruntime使用gpu推理-20220402115703](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-04-02-onnxruntime使用gpu推理-20220402115703.png)

上面的绿色的cuda是一个软链接，指向的是cuda-11.3这个目录。cuda-11.3是从nvidia官网上下载的。
创建软链接的操作：

sudo ln -s /usr/local/cuda/ /usr/local/cuda-11.3/

除此之外，还需要从官网下载cudnn，将cudnn放入cuda-11.3中。

假设下载后的cudnn，解压缩后的目录是：folder/extracted/contents 
那么执行：
```bash
cd folder/extracted/contents 

cp include/cudnn.h /usr/local/cuda/include

cp lib64/libcudnn* /usr/local/cuda/lib64 
 
chmod a+r /usr/local/cuda/lib64/libcudnn* 
```

需要注意：onnxruntme、cuda和cudnn之间的版本要对应，具体可以从这个网址查看：

[https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)


# 4、设置PATH、LD_LIBRARY_PATH

这是让onnxruntime找到cuda的关键一步。
操作如下：
```bash
export PATH=/usr/local/cuda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
可以将上面的操作放到~/.bashrc中，然后使用source更新一下。


至此，运行onnxruntime推理时，应该可以从nvidia-smi中看到gpu被使用到了。


















