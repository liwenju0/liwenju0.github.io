---
layout: post
title:  "python中的keyword argument"
date:   2022-02-23 10:04:08 +0800
category: "计算机基础"
published: true
---

keyword arguments有两个含义。

## 一、函数调用时的keyword arguments

这里的含义是说，你可以在调用函数时，通过key=value这种方式，指定某个参数的值。这里，你不用关心这个参数是不是positional arguments，以及有没有默认值。

唯一的要求是，这些key=value要在没有名字的positional arguments后面。

<!--more-->


## 二、函数定义时的keyword arguments

这里的含义是，你可以在定义函数时，让函数能接受带名字的参数，但是你并不需要指明这些名字。一般通过**kwargs进行明确。为了区别这些参数和arg2=0这类含有默认值的positional arguments。可以称它为纯keyword arguments。

## 三、小结

函数的参数，按照定义的顺序，分成如下四类：

- 没有默认值的positional arguments
- 有默认值的positional arguments
- 多余的 positional arguments  即*args
- 纯的keyword arguments       即 **kwargs

第一类参数，在函数调用时，必须通过positional arguments或者key=value的形式之一进行明确，且只能使用其中一个方法。

第二类参数，在函数调用时，可以通过positional arguments或者key=value的形式之一进行明确，如果两种方式都没有明确，则使用默认值。

在函数调用时的positional arguments，它是严格按照顺序对应到函数定义时的positional arguments。谨记！


