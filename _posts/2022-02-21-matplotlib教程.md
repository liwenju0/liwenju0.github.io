---
layout: post
title:  "matplotlib教程"
date:   2022-02-21 10:04:08 +0800
tags: python
---

matplotlib是优秀的python画图工具，功能十分强大，但是使用却很复杂。你有没有如下的经历:

1、图形只差一点点就满足你的要求，可是怎么调 也调不到位

2、好不容易从stackoverflow上查到一个解决方案 ，可使用时却各种调整无法达到预期，或者好不容易搞定了。随便换个图又不好使了

3、网上一下查到好几个方案，不知道到底哪个好，只能一个一个试

4、有时候，想要调整一个地方，可是不知道怎么搜索关键字

如果你有过以上的经历，恭喜，这个教程就是为你量身定做的。这个教程和其他教程有啥区别？答案是：这个教程是从架构的高度来讲解matplotlib的，学完后，你不只是知道了怎么使用matplotlib，更是知道为什么要这样使用。当你脑子中有一个图的模样时，你知道如何组合不同的matplotlib的功能来实现它。


# matplotlib的编程范式
matplotlib同时支持过程式和面向对象式的使用方法。也就是同一个效果，大部分情况下都有至少两种方法实现。这给初学者造成了很多困惑。所以，我们要先了解这两种方式。

首先看一下过程式的方法：

```python
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
```
这段代码的效果如下：

![2022-02-21-matplotlib教程-20220221170817](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221170817.png)

这种风格是典型的matlab风格，也叫基于状态的绘制方法。

另一种就是官方推荐的面向对象的绘制方法。上面的图，用面向对象来来写，就是这样的：
```python
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
fig, (ax1, ax2) = plt.subplots(2,1)


ax1.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

ax2.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
```

这种风格的好处是绘制对象非常明确。我们这门课程主要是用面向对象的风格，这个也符合matplotlib的最佳实践！


我们要学习的是matplotlib中的核心概念，这些概念对学会matplotlib至关重要。本文会循序渐进地讲清楚每一个概念。


# figure
matplotlib中最大的概念就是figure，一个figure就是一幅图，可以把它理解成一个有大小的画布。
![2022-02-21-matplotlib教程-20220221171208](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171208.png)

那么，下面的问题自然是：如何获得一个figure？如何在figure上画图？首先，我们来看看获得一个figure的办法：
```python
fig = plt.figure()
```
这样就获得了一个figure。在解决如何在figure上画图之前，我们先来观察一下这个fig。既然figure是画布，那么大小如何设置呢？可以这样来设置：
```python
fig.set_figheight(100)
fig.set_figwidth(100)
```
看起来比较直观，但是这里有个巨坑的地方，就是里面的100代表的是英寸！！！如果我想要一个(1200,600)像素的图，该怎么办呢？要达成目标，我们必须了解figure的另一个属性:dpi(dot per inch)。它代表的是每英寸有多少个像素点。默认值是72。我们可以使用如下的三种设置中的一种得到(1200,600)像素的图：

```python
figsize=(宽，高)
figsize=(15,7.5), dpi= 80
figsize=(12,6)  , dpi=100
figsize=( 8,4)  , dpi=150
```
看，光是得到一个大小确定的figure就这么麻烦。但这正是matplotlib厉害的地方，高度的可定制性。seaborn等绘图库都只是对matplotlib的封装而已。学会matplotlib，其他的绘图库可以很快学会。

既然fig代表一幅图，如果我们还想继续画一幅图怎么办呢？看如下代码：

```python
fig1 = plt.figure()
fig2 = plt.figure()
```
这里，fig1和fig2代表两个不同的图，这可以通过它们的number属性看出来：

![2022-02-21-matplotlib教程-20220221171238](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171238.png)


matplotlib内部维护着一个全局的计数，记录了一共创建了多少个figure。

# figure的属性
现在我们来总整体上看看figure有哪些属性，执行fig.__dict__可以得到如下内容：

```python
{'_stale': True,
 'stale_callback': None,
 'figure': None,
 '_transform': None,
 '_transformSet': False,
 '_visible': True,
 '_animated': False,
 '_alpha': None,
 'clipbox': None,
 '_clippath': None,
 '_clipon': True,
 '_label': '',
 '_picker': None,
 '_contains': None,
 '_rasterized': None,
 '_agg_filter': None,
 '_mouseover': False,
 'eventson': False,
 '_oid': 0,
 '_propobservers': {},
 '_remove_method': None,
 '_url': None,
 '_gid': None,
 '_snap': None,
 '_sketch': None,
 '_path_effects': [],
 '_sticky_edges': _XYPair(x=[], y=[]),
 '_in_layout': True,
 'callbacks': <matplotlib.cbook.CallbackRegistry at 0x7f75632e4410>,
 'bbox_inches': Bbox([[0.0, 0.0], [6.0, 4.0]]),
 'dpi_scale_trans': <matplotlib.transforms.Affine2D at 0x7f75655edc90>,
 '_dpi': 72.0,
 'bbox': <matplotlib.transforms.TransformedBbox at 0x7f75655ed0d0>,
 'transFigure': <matplotlib.transforms.BboxTransformTo at 0x7f75655ed090>,
 'patch': <matplotlib.patches.Rectangle at 0x7f7563004310>,
 'canvas': <matplotlib.backends.backend_agg.FigureCanvasAgg at 0x7f75632e4e10>,
 '_suptitle': None,
 'subplotpars': <matplotlib.figure.SubplotParams at 0x7f75632e4e50>,
 '_layoutbox': None,
 '_constrained_layout_pads': {'w_pad': 0.04167,
  'h_pad': 0.04167,
  'wspace': 0.02,
  'hspace': 0.02},
 '_constrained': False,
 '_tight': False,
 '_tight_parameters': {},
 '_axstack': <matplotlib.figure.AxesStack at 0x7f75632e4ed0>,
 'suppressComposite': None,
 'artists': [],
 'lines': [],
 'patches': [],
 'texts': [],
 'images': [],
 'legends': [],
 '_axobservers': [<function matplotlib.backend_bases.FigureManagerBase.__init__.<locals>.notify_axes_change(fig)>],
 '_cachedRenderer': None,
 '_align_xlabel_grp': <matplotlib.cbook.Grouper at 0x7f75655edd90>,
 '_align_ylabel_grp': <matplotlib.cbook.Grouper at 0x7f75632e4090>,
 '_gridspecs': [],
 'number': 1}
```
属性非常多，主要分为两类，一类是对figure的设置，比如wspace、hspace等。另一类是和figure中画的内容有关，比如lines、texts、legends、artists等。

此时不需要理解全部的属性，只要浏览一遍，有个大致的印象就可以了。




# Axes和subplot
开局一张图：
![2022-02-21-matplotlib教程-20220221171500](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171500.png)

上一讲中，我们获得了一个figure，并且也大致了解了一下figure的相关设置项。

现在，我们要在figure上画图了。你本来以为直接在figure上画就行了，但事实没这么简单。matplotlib允许我们将一个figure通过栅格系统划分成不同的格子，然后在格子中画图，这样就可以在一个figure中画多个图了。这里的每个格子有两个名称：Axes和subplot。subplot是从figure所有的格子来看的。因为figure要统一管理协调这些格子的位置、间隔等属性，管理协调的方法和属性设置就在subplots的层面进行。

Axes是从作为画图者的我们的角度来定义的，我们要画的点、线等都在Axes这个层面来进行。画图用的坐标系统自然也是在Axes中来设置的。

搞清楚这两个概念后，我们就来看看如何将figure划分格子，并获得我们画图使用的Axes。
```python
axes = fig.subplots(2,2)
```
以上代码，我们将整个fig划分成了2x2 4个subplots。返回给我们的自然是四个axes，可以通过查看axes证实：
![2022-02-21-matplotlib教程-20220221171613](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171613.png)

在上图里，我们看到返回的对象是AxesSubplot，它实质上是包含了Axes的Subplot。在使用上，我们完全可以把它当做Axes使用。

如果我们只想在figure上画一幅图，就有两种方法：

```python
axes = fig3.subplots(1,1)
# or
axes = fig3.subplots()
```
此时得到的axes是就是一个AxesSubplot对象。
![2022-02-21-matplotlib教程-20220221171749](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171749.png)

如果大家观察仔细，会看到里面有3个值，它们确定了subplot在figure中的位置。可以通过下图感受到：

![2022-02-21-matplotlib教程-20220221171812](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171812.png)
前两个值实际上是坐标原点相对于figure左下角的位置。第三个值是subplot的宽和高。

figure中还有一个方法：add_subplot。其目的也是将figure划分成栅格，并获取其中某一个。使用方法如下所示：
```python
fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)  
fig.add_subplot(232, facecolor="blue")  
fig.add_subplot(233, facecolor="yellow")  
fig.add_subplot(234, sharex=ax1) 
fig.add_subplot(235, facecolor="red") 
fig.add_subplot(236, facecolor="green")  
plt.show()
```

输出如下图：

![2022-02-21-matplotlib教程-20220221171930](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221171930.png)

这里有两个地方需要注意一下。add_subplot(232)和add_subplot(2,3,2)等价的。

另外，如果将最后一个236改成436，你猜会发生什么呢？

答案是如下所示：

![2022-02-21-matplotlib教程-20220221172211](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172211.png)

可以看到436 相当于将figure重新划分了网格，并将第6个网格设置成绿色。

两种不同的网格划分产生了重叠。这再次体现了matplotlib的灵活性。

最佳的实践是：在开始时就将figure的网格划分好，并不再改变。

# Axes 概览
最后，我们还是通览一下Axes的属性：
```python
{'figure': <Figure size 432x288 with1 Axes>,
 '_subplotspec': <matplotlib.gridspec.SubplotSpec at 0x7f039a304bd0>,
 'figbox': Bbox([[0.125, 0.125], [0.9, 0.88]]),
 'rowNum': 0,
 'colNum': 0,
 'numRows': 1,
 'numCols': 1,
 '_stale': True,
 'stale_callback': <function matplotlib.figure._stale_figure_callback(self, val)>,
 '_axes': <matplotlib.axes._subplots.AxesSubplot at 0x7f039b097f90>,
 '_transform': None,
 '_transformSet': False,
 '_visible': True,
 '_animated': False,
 '_alpha': None,
 'clipbox': None,
 '_clippath': None,
 '_clipon': True,
 '_label': '',
 '_picker': None,
 '_contains': None,
 '_rasterized': None,
 '_agg_filter': None,
 '_mouseover': False,
 'eventson': False,
 '_oid': 0,
 '_propobservers': {},
 '_remove_method': <bound method Figure._remove_ax of <Figure size 432x288 with1 Axes>>,
 '_url': None,
 '_gid': None,
 '_snap': None,
 '_sketch': None,
 '_path_effects': [],
 '_sticky_edges': _XYPair(x=[], y=[]),
 '_in_layout': True,
 '_position': Bbox([[0.125, 0.125], [0.9, 0.88]]),
 '_originalPosition': Bbox([[0.125, 0.125], [0.9, 0.88]]),
 '_aspect': 'auto',
 '_adjustable': 'box',
 '_anchor': 'C',
 '_sharex': None,
 '_sharey': None,
 'bbox': <matplotlib.transforms.TransformedBbox at 0x7f039a304a10>,
 'dataLim': Bbox([[inf, inf], [-inf, -inf]]),
 'viewLim': Bbox([[0.0, 0.0], [1.0, 1.0]]),
 'transScale': <matplotlib.transforms.TransformWrapper at 0x7f0399f6ae90>,
 'transAxes': <matplotlib.transforms.BboxTransformTo at 0x7f0399f6a690>,
 'transLimits': <matplotlib.transforms.BboxTransformFrom at 0x7f0399f6a410>,
 'transData': <matplotlib.transforms.CompositeGenericTransform at 0x7f039b091ed0>,
 '_xaxis_transform': <matplotlib.transforms.BlendedGenericTransform at 0x7f039b091310>,
 '_yaxis_transform': <matplotlib.transforms.BlendedGenericTransform at 0x7f039b091610>,
 '_axes_locator': None,
 'spines': OrderedDict([('left', <matplotlib.spines.Spine at 0x7f039ac36050>),
              ('right', <matplotlib.spines.Spine at 0x7f039ac368d0>),
              ('bottom', <matplotlib.spines.Spine at 0x7f039ac36410>),
              ('top', <matplotlib.spines.Spine at 0x7f039ac36810>)]),
 'xaxis': <matplotlib.axis.XAxis at 0x7f039b0913d0>,
 'yaxis': <matplotlib.axis.YAxis at 0x7f039b146510>,
 '_facecolor': 'white',
 '_frameon': True,
 '_axisbelow': 'line',
 '_rasterization_zorder': None,
 '_connected': {},
 'ignore_existing_data_limits': True,
 'callbacks': <matplotlib.cbook.CallbackRegistry at 0x7f039ac36cd0>,
 '_autoscaleXon': True,
 '_autoscaleYon': True,
 '_xmargin': 0.05,
 '_ymargin': 0.05,
 '_tight': None,
 '_use_sticky_edges': True,
 '_get_lines': <matplotlib.axes._base._process_plot_var_args at 0x7f039a2e9850>,
 '_get_patches_for_fill': <matplotlib.axes._base._process_plot_var_args at 0x7f039a58a290>,
 '_gridOn': False,
 'lines': [],
 'patches': [],
 'texts': [],
 'tables': [],
 'artists': [],
 'images': [],
 '_mouseover_set': <matplotlib.cbook._OrderedSet at 0x7f039b02c190>,
 'child_axes': [],
 '_current_image': None,
 'legend_': None,
 'collections': [],
 'containers': [],
 'title': Text(0.5, 1, ''),
 '_left_title': Text(0.0, 1, ''),
 '_right_title': Text(1.0, 1, ''),
 'titleOffsetTrans': <matplotlib.transforms.ScaledTranslation at 0x7f039b80a450>,
 '_autotitlepos': True,
 'patch': <matplotlib.patches.Rectangle at 0x7f039b80a590>,
 'axison': True,
 'fmt_xdata': None,
 'fmt_ydata': None,
 '_navigate': True,
 '_navigate_mode': None,
 '_xcid': 0,
 '_ycid': 0,
 '_layoutbox': None,
 '_poslayoutbox': None}
```
可以看到，里面有Axes在网格系统中的坐标，所属的figure，还有title，_facecolor等属性。yaxis、xaxis代表坐标轴。

此时，建议大家对比一下Axes和上一讲中的Figure的属性。就可以大致有个感觉，什么样的设置要在哪里去设。

今天内容先到这里吧。这几篇文章讲的知识点都比较少，是因为这些知识点是整个matplotlib大厦的根基。只有充分地理解了这些知识点，后面的学习才会更加容易。


在前面的文章中，我们已经了解到Axes才是我们绘图的主战场。今天我们就来看看Axes中如何进行绘图。

# Axes中的各种对象
本系列教程的开篇中，我们就了解到，matplotlib有过程式和面向对象式两种使用方法。官方推荐的最佳实践是使用面向对象的方式。

同样在画图时，matplotlib是把各种元素也按照对象进行组织的。下面的图展示了一个图中，各种组件对应的对象名称：

![2022-02-21-matplotlib教程-20220221172351](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172351.png)

熟悉这个图里的各个组件的名字至关重要哦。因为以后要设置某个部分，你首先需要先了解各个部分的名称。


# Artist

上面各种组件都是视觉可见的。为了有统一的层次结构，matplotlib给所有视觉可见的组件定义了一个统一的基类:Artist。整个matplotlib中的可见对象如下所示：

![2022-02-21-matplotlib教程-20220221172434](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172434.png)


这幅图虽然很庞大，不要紧，现在先将精力集中在看的懂的组件上就可以了。从整体上看，共有两类Artist，我们先看图再解释：

![2022-02-21-matplotlib教程-20220221172504](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172504.png)

containers就是容器，能够容纳其他的Artist的Artist。比如Axes、Figure都是containers。另一类就是基本图，即primitives，如线、图、文字等。

容器中可以有各种各样的Artists，为了便于管理，会为每一类primitive创建一个列表。在上一篇文章中，可以看到Axes中有lines、artists、images等列表。

# 四种常见的容器

Figure，Axes、Axis、Tick是常见的四种容器，每种容器的属性我们最好熟悉一下，列到下面供参考：


![2022-02-21-matplotlib教程-20220221172613](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172613.png)


![2022-02-21-matplotlib教程-20220221172625](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172625.png)

![2022-02-21-matplotlib教程-20220221172635](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172635.png)

![2022-02-21-matplotlib教程-20220221172645](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172645.png)

好了，通过前面的4讲，我们主要是理清了matplotlib中最重要的基本概念。这样的做法，和你见到的大多数matplotlib教程很不一样。原因是我觉得这样才是正确的学习方法。学完这些概念，你会发现，当你看到一个图不符合预期的时候，你知道应该调整哪里，或者查找哪个关键词，再也不会一头雾水了。

后面的教程中，我们会开始具体讲解各种绘图组件了。

今天我们的目标是学习常用的图形绘制，经过前面的铺垫，现在再来学习这些图形的绘制，就非常的简单了。

# plot
这是最简单的图，也就是折线。如下代码所示：
```python
fig = plt.figure()
ax = fig.subplots()
x = np.arange(0, np.pi*2, 0.05)
y = np.sin(x)
ax.plot(x, y)

ax.set(xlim=[0, 3.5], ylim=[0, 1], title='An Example Axes',
       ylabel='Y-Axis', xlabel='X-Axis')
plt.show()
```

图形如下：
![2022-02-21-matplotlib教程-20220221172817](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172817.png)

通过图形可以看到，xlim、ylim，title、ylable、xlabel这些都是在Axes中进行设置的，学习完前面的知识，你会感觉这样的安排是很自然的。

同时，针对每一个设置，Axes都有单独的set方法，以方便我们的使用。
```python
ax.set_xlim([0, 3.5])
ax.set_ylim([0, 1])
ax.set_title('A Different Example Axes Title')
ax.set_ylabel('Y-Axis (changed)')
ax.set_xlabel('X-Axis (changed)')
```

这种单独设置的好处在于，可以针对每一个设置项进行更细粒度的设置，比如如果想设置title的字体大小，并设置title的位置。可以这样：
```python
ax.set_title('A Title',
fontdict={"fontsize":20, "color":"blue"},
loc = "right",
)
```
如下图所示：
![2022-02-21-matplotlib教程-20220221172907](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172907.png)


如果要设置线的颜色，可以在plot方法中直接设置：
```python
ax.plot(..., color = "green", ...)
```
你问我，都有哪些颜色可用呢？都在这里了：
![2022-02-21-matplotlib教程-20220221172936](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221172936.png)

线型也可以在plot方法中直接设置。
```python
ax.plot(..., linestyle='--', ...)
```
matplotlib支持如下线型：
![2022-02-21-matplotlib教程-20220221173009](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221173009.png)

# scatter
散点图，直接看代码：
```python
import math
fig = plt.figure()
ax = fig.subplots()
x = [0,2,4,6,8,10,12,14,16,18]
s_exp = [20*2**n for n in range(len(x))]
s_square = [20*n**2for n in range(len(x))]
s_linear = [20*n for n in range(len(x))]

ax.scatter(x,[1]*len(x),s=s_exp, label='$s=2^n$', lw=1)
ax.scatter(x,[0]*len(x),s=s_square, label='$s=n^2$')
ax.scatter(x,[-1]*len(x),s=s_linear, label='$s=n$')
ax.set_ylim([-1.5,1.5])
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), labelspacing=3)
plt.show()
```
图如下：
![2022-02-21-matplotlib教程-20220221173141](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221173141.png)

上面的代码中，我们看到：

matplotlib允许我们控制每一个点的大小，这是通过scatter中的s属性确定的。

label属性的作用是，当一个Axes中有多个图时，用来标记在图例中，比较厉害的是，这里允许使用latex语法，再次体现了matplotlib的强大。

# legend
legend的位置确定是个重要却有点复杂的设置，我们单拎出来说说。matplotlib确定legend的位置实际上有两套逻辑，而且两套逻辑同时用到 loc 和 bbox_to_anchor。这是造成混乱的根本原因。

## 1、第一套逻辑

先用bbox_to_anchor确定一个方框，loc是legend在这个方框中的位置。

确定方框的bbox_to_anchor需要是一个四元组(x,y,width, height)。里面的数值都是相对Axes的比例坐标。比如其默认值是(0,0,1,1)，代表的是整个Axes。(0,0,0.5,0.5)代表的是Axes的左下四分之一的矩形框。

loc是legend在这个方框中的位置，可以使用的位置如下所示：

![2022-02-21-matplotlib教程-20220221173246](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221173246.png)
## 2、第二套逻辑

这套逻辑是先用bbox_to_anchor确定一个点，然后loc表示的是这个点相对legend的位置。这个逻辑就有点绕了。我们可以通过图来感受一下。

bbox_to_anchor是(0.6,0.5)时，即下图中的绿点位置，不同的loc确定的legend的位置：

![2022-02-21-matplotlib教程-20220221173316](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-matplotlib教程-20220221173316.png)
只要理解了这两套逻辑，就不会混乱了。


