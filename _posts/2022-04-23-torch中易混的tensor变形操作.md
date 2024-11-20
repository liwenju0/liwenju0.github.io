---
layout: post
title:  "torch中易混的tensor变形操作"
date:   2022-04-23 09:20:08 +0800
category: "AI"
published: true
---

这四对相似又不完全一样的api的简要解析。


<!--more-->

# shape和size
shape就是size的别称。

t是一个tensor的话。

t.shape 和t.size()是等价的。

# expand和repeat

从反向传播的角度看，二者是等价的。

不同之处时，当要扩展的原始维度是1时，可以使用expand来免于内存复制，且expand也只能对尺寸为1的维度进行扩展。
当维度尺寸大于1时，只能使用repeat。

# view和reshape
view只能在contiguous的tensor上操作，reshape没有这个限制。

view返回的tensor和原始tensor共享底层数据。

reshape在可能的情况下，返回的tensor也和原始tensor共享底层数据，当不可能时，会进行复制。

# permute和transpose
permute可以改变tensor各个维度的顺序。

transpose是2D的tensor情况下特殊的permute，因为此时只有一种permute方式。