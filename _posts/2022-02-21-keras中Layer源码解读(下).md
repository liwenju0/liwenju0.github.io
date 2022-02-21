---
layout: post
title:  "keras中Layer源码解读（下）"
date:   2022-02-21 10:04:08 +0800
tags: 深度学习
---

今天继续读keras中的Layer源码

# 175-191
![2022-02-21-keras中Layer源码解读(下)-20220221175937](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221175937.png)

这是一个静态方法，注释中说，在self._network_nodes中内部使用。可是搜不到这个属性。大概就是为Layer生成一个唯一的名字。查找了该方法的应用。主要是在Network这个类中。

于是首先查看了一下Network的说明：
![2022-02-21-keras中Layer源码解读(下)-20220221175958](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221175958.png)

原来，Network是layers组成的有向无环图，Model不过是添加了训练方法的Network。

在其中找到一个使用_node_key的地方看看：

![2022-02-21-keras中Layer源码解读(下)-20220221180015](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180015.png)

可以看到，确实是用来给Network中的各个layer生成唯一的名字，好方便Network对layers的组织。


# 192-198

![2022-02-21-keras中Layer源码解读(下)-20220221180035](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180035.png)

汇总各个layer的loss。这里学习到python的一个小技巧。list的 [:]操作，相当于对list的复制。之前都没注意过。

下面直到249行，内容都是一些getter和setter。

# 250-292
![2022-02-21-keras中Layer源码解读(下)-20220221180102](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180102.png)


这个add_weight方法是keras目前推荐的做法。这里看到一个python的知识点，contextmanager。进入K.name_scope方法中。

![2022-02-21-keras中Layer源码解读(下)-20220221180128](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180128.png)

简单理解，@contextmanager的作用是使该方法可以通过 with语法使用。

看方法里面，实现的相当简单，就是在一个全局的数组中，添加name。然后yield。此时，with区块内，进行变量命名时，会使用NAME_SCOPE_STACK中的所有元素，自然包含了刚添加进来的name。

当退出with区块后，再将该name pop出来。

# 293-400
这个方法是assert_input_compatibility，值得认真学习一下。因为我们自己创建自己的模型时，经常需要检查不同的tensor之间的shape是否适配。希望能通过这个方法，学到一些小技巧。

```python
def assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        inputs = to_list(inputs)
        for x in inputs:
            try:
                K.is_keras_tensor(x)
            except ValueError:
                raise ValueError('Layer ' + self.name + ' was called with '
                                 'an input that isn\'t a symbolic tensor. '
                                 'Received type: ' +
                                 str(type(x)) + '. Full input: ' +
                                 str(inputs) + '. All inputs to the layer '
                                 'should be tensors.')

        ifnot self.input_spec:
            return
        ifnot isinstance(self.input_spec, (list, tuple)):
            input_spec = to_list(self.input_spec)
        else:
            input_spec = self.input_spec
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                             'but it received ' + str(len(inputs)) +
                             ' input tensors. Input received: ' +
                             str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec isNone:
                continue

            # Check ndim.
            if spec.ndim isnotNone:
                if K.ndim(x) != spec.ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.max_ndim isnotNone:
                ndim = K.ndim(x)
                if ndim isnotNoneand ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.min_ndim isnotNone:
                ndim = K.ndim(x)
                if ndim isnotNoneand ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            # Check dtype.
            if spec.dtype isnotNone:
                if K.dtype(x) != spec.dtype:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected dtype=' +
                                     str(spec.dtype) + ', found dtype=' +
                                     str(K.dtype(x)))
            # Check specific shape axes.
            if spec.axes:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape isnotNone:
                    for axis, value in spec.axes.items():
                        if (value isnotNoneand
                                x_shape[int(axis)] notin {value, None}):
                            raise ValueError(
                                'Input ' + str(input_index) +
                                ' is incompatible with layer ' +
                                self.name + ': expected axis ' +
                                str(axis) + ' of input shape to have '
                                'value ' + str(value) +
                                ' but got shape ' + str(x_shape))
            # Check shape.
            if spec.shape isnotNone:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape isnotNone:
                    for spec_dim, dim in zip(spec.shape, x_shape):
                        if spec_dim isnotNoneand dim isnotNone:
                            if spec_dim != dim:
                                raise ValueError(
                                    'Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected shape=' +
                                    str(spec.shape) + ', found shape=' +
                                    str(x_shape))

```

首先学到的一点，这个方法，在检测出问题后，不是返回False，而是直接抛出异常。这是合理的。因为shape不兼容，后面的运算就没有意义，及时抛出异常，让用户检查是最合理的。

首先，检查每一个input是否是tensor，不是的话，异常爆出！具体是怎么检查的呢？我有点好奇。

于是点进K.is_keras_tensor方法，看到了keras tensor的定义：

![2022-02-21-keras中Layer源码解读(下)-20220221180237](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180237.png)

前两天就遇到这个问题，一个tensor不是keras tensor，结果放到模型中时各种报错。如果用这个方法检查一下，就会安全很多。

![2022-02-21-keras中Layer源码解读(下)-20220221180257](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180257.png)

这里看到，keras tensor 首先要是tensor，然后有_keras_history属性。

要是tensor，就要是tf_ops._TensorLike或者tf_ops.is_dense_tensor_like(x)为真。

_keras_history属性，是干什么的？查找一下源代码，在_add_inbound_node中看到如下代码：

![2022-02-21-keras中Layer源码解读(下)-20220221180311](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180311.png)

可见，_keras_history就是一个三元组，这个三元组就是相当于tensor的坐标。通过下面的代码就可以看出来：

![2022-02-21-keras中Layer源码解读(下)-20220221180331](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180331.png)

第一个表示这个产生这个tensor的layer，第二个表示该tensor在layer的inbound_nodes数组中对应的Node的index，第三个表示该tensor是该layer对应Node中的output_tensors数组中的index。

再简单说，就是在keras中，一个tensor，可以先找到产生它的layer，然后在该layer的inbound_nodes中找到产生它的Node，然后在Node的output_tensors中找到它。这就是tensor坐标的含义。

不知不觉，文章已经够长了。下回继续吧。


今天继续读keras中的Layer源码。

# 293-400
```python
def assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        inputs = to_list(inputs)
        for x in inputs:
            try:
                K.is_keras_tensor(x)
            except ValueError:
                raise ValueError('Layer ' + self.name + ' was called with '
                                 'an input that isn\'t a symbolic tensor. '
                                 'Received type: ' +
                                 str(type(x)) + '. Full input: ' +
                                 str(inputs) + '. All inputs to the layer '
                                 'should be tensors.')

        ifnot self.input_spec:
            return
        ifnot isinstance(self.input_spec, (list, tuple)):
            input_spec = to_list(self.input_spec)
        else:
            input_spec = self.input_spec
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                             'but it received ' + str(len(inputs)) +
                             ' input tensors. Input received: ' +
                             str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec isNone:
                continue

            # Check ndim.
            if spec.ndim isnotNone:
                if K.ndim(x) != spec.ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.max_ndim isnotNone:
                ndim = K.ndim(x)
                if ndim isnotNoneand ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.min_ndim isnotNone:
                ndim = K.ndim(x)
                if ndim isnotNoneand ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            # Check dtype.
            if spec.dtype isnotNone:
                if K.dtype(x) != spec.dtype:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected dtype=' +
                                     str(spec.dtype) + ', found dtype=' +
                                     str(K.dtype(x)))
            # Check specific shape axes.
            if spec.axes:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape isnotNone:
                    for axis, value in spec.axes.items():
                        if (value isnotNoneand
                                x_shape[int(axis)] notin {value, None}):
                            raise ValueError(
                                'Input ' + str(input_index) +
                                ' is incompatible with layer ' +
                                self.name + ': expected axis ' +
                                str(axis) + ' of input shape to have '
                                'value ' + str(value) +
                                ' but got shape ' + str(x_shape))
            # Check shape.
            if spec.shape isnotNone:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape isnotNone:
                    for spec_dim, dim in zip(spec.shape, x_shape):
                        if spec_dim isnotNoneand dim isnotNone:
                            if spec_dim != dim:
                                raise ValueError(
                                    'Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected shape=' +
                                    str(spec.shape) + ', found shape=' +
                                    str(x_shape))

```
在assert_input_compatibility中，检查完tensor后，进入shape的检查。每一个tensor对应一个input_spec，如果对应不上，就会报错。

看到首先检查的是tensor的维度数，即ndim。其次是 max_ndim和min_ndim。

维度检查完毕，就开始检查数据类型。360-366

继续，检查每一个axis的维度数。368-383 这里的检查是比较宽松的，因为允许 维度是None。

紧接着的shape检查就会要求二者严格相等了。385-399

看起来，spec.shape和spec.axes更像是在版本迭代中重叠起来的。为了兼容，一直保留了下来。

# 401-412
![2022-02-21-keras中Layer源码解读(下)-20220221180450](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180450.png)

这里是call方法，留给具体的Layer进行实现。其实就是模板设计模式。我们自定义自己的Layer时，通过重写这个方法实现自己的计算逻辑。

# 413-540
这里是call方法的包装方法，用来处理预处理一些keras的记录信息。

![2022-02-21-keras中Layer源码解读(下)-20220221180529](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180529.png)


读一下注释，可以发现这个包装方法主要完成四项功能：

- 调用_add_inbound_node()方法
  
这里想进一步理解一下inbound_node这个概念。我们的layer创建好后，可以多次调用，如下所示：
```python
myLayer = MyLayer()
o1 = myLayer(inputs1)
o2 = myLayer(inputs2)
```

inputs1和inputs2是input tensor的列表，里面的每个input都来自某一个上游的Layer的output。但是每个列表中的input未必来自同一个Layer的output。

在上面，由于我们调用了两次myLayer。所以，在myLayer的inbound_nodes数组中就会添加两个node，来分别记录这两次调用的信息。具体记录哪些信息，就在这个包装的__call__方法中。

- 如果layer还没有built，会进行build方法的调用

- 更新_keras_shape，这个需要结合代码进行理解

- 对每一个output tensor, 更新它的_keras_history。这个在上篇文章中已经讲过。每当产生一个新的tensor，及时确定它的三维坐标（layer，node_index，output_tensor_index)

这段注释，我觉得写的非常好。值得学习。提纲挈领地说明了这个方法的内容。这样，在下面的阅读中，就不会迷失在浩繁的细节中。
![2022-02-21-keras中Layer源码解读(下)-20220221180651](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180651.png)

这里，首先将inputs进行复制。

然后加入了name_scope，因为可能需要调用build方法，添加weight，这样，每个weight就会在该layer的name_scope下了。

name_scope这个概念，我觉得直接理解成命名前缀就可以了。简单易懂。

调用built前，先检查了inputs的各个shape，这个方法我们刚学习过。

通过后，开始依次收集每个input的shape，用来进行build的调用。

获取shape的方法值得学习一下：

![2022-02-21-keras中Layer源码解读(下)-20220221180708](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180708.png)

看到调用了build。然后将built置为True。

这里，build方法如下：
![2022-02-21-keras中Layer源码解读(下)-20220221180725](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180725.png)

这也是一个模板方法，如果我们自定义的layer中有weight，就需要重写这个方法，在里面添加我们的weight。感觉这种模板设计特别清晰容易理解。

如果没有build方法，那么自定义layer时就可能在任何地方添加我们的weight，这样肯定会引发混乱，不利于大家交流代码。所以，这个build方法其实就是一个约定。

build完之后，初始化了weights，并再次检查了input shape，因为在build方法中，可能会创建input_spec。这些我们都比较熟悉了。

再往下，就是mask的处理了。这块儿，是比较容易搞乱的地方。需要认真研究一下。

# 475

![2022-02-21-keras中Layer源码解读(下)-20220221180749](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180749.png)


看一下，_collect_previous_mask这个方法：

![2022-02-21-keras中Layer源码解读(下)-20220221180806](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180806.png)

可以看到，首先获取了这个tensor的坐标，然后，找到了对应的node，在node的output_masks中根据tensor_index获取到mask，添加进结果list中。

这里，进一步可以看到Node的作用。存储了每个output tensor的mask。细心体会一下，就能感觉到Node这层抽象的必要性。因为像mask这种信息，显然不适合直接放到layer中。

![2022-02-21-keras中Layer源码解读(下)-20220221180822](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180822.png)

这里看到，收集了mask信息后，添加到kwargs中，当然，如果你明确传入了mask，就不会用收集到的mask更新。前提是，你的call方法中还要有mask这个参数。这个参数表明了你的layer是support_mask的。

至此，就会实际调用我们自己定义的逻辑。

![2022-02-21-keras中Layer源码解读(下)-20220221180837](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180837.png)

需要研究一下，是怎么计算mask的。

![2022-02-21-keras中Layer源码解读(下)-20220221180856](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180856.png)

这里的mask就是简单传递一下。为了找到一个具体例子，我想起来了Embedding这个layer，找到里面的方法：



![2022-02-21-keras中Layer源码解读(下)-20220221180905](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221180905.png)

可以看到，计算是相当直接。就是将等于0的值进行mask。如果你想用其他的值进行mask，就可以继承这个layer，重写相关的方法。

今天内容不少了。先到这里吧。

继续keras源码阅读，不急不躁，慢就是快。

# 490-502
这里处理了一个很trick的问题。就是当layer只是简单将input返回作为output时，要对output进行一下复制，防止丢失tensor的元数据。

复制的函数就是 K.identify。

这个函数的签名说的很清楚：
![2022-02-21-keras中Layer源码解读(下)-20220221181006](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181006.png)

复制一个tensor的值，感觉这个函数还是很有用的。值得用心记一下。

# 503-512
这段代码不用考虑，因为注释说的很明白：
![2022-02-21-keras中Layer源码解读(下)-20220221181028](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181028.png)

# 513-517
![2022-02-21-keras中Layer源码解读(下)-20220221181050](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181050.png)

这段同样是对mask进行处理。上一篇文章中，看到Embedding layer计算mask时，根据input是否等于0，产生了mask。

但是有一个问题，就是当输出是多个tensor时，需要每个tensor有一个mask。这段代码就是做这件事儿的。为什么不用一个呢？

我想这样的设计是为了简单，清晰。每个output tensor都对应自己的mask。数据结构非常整齐。

# 518-530
![2022-02-21-keras中Layer源码解读(下)-20220221181113](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181113.png)

又到了这段代码，在前面已经见过几次了。不过我觉得还是有必要认认真真读一下。因为它对理解keras组织神经网络的方式非常重要。

首先就是注释。说明了Node的作用。

1、追踪对该layer的call调用

2、追踪call调用产生的新的variable

3、更新output tensor的keras history

这一点我们之前已经学习过，就是tensor的坐标。强调一点，我觉得tensor坐标是一个非常重要的概念。它揭示出了keras中很多的设计初衷。

4、如果input tensor有自己的坐标，则也进行记录。

看完注释，自然是要进到方法内部去看看。

![2022-02-21-keras中Layer源码解读(下)-20220221181134](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181134.png)

这是上面讲的第4点的内容。记录input tensor的坐标。

inbound_layers = []

node_indices = []

tensor_indices = []

就是这三项。当然，前提是input tensor 要有_keras_history这个属性，如果没有，就用None代替。

看到这里，我想到，keras现在强绑定tensorflow，那按道理，现在tensorflow的每个tensor应该都有_keras_history了吧。为了验证，我打开了tensorflow的源码。发现并没有。

这样就理解了一个限制，keras layer的输入必须是其他的keras的layer的输出，目的就是确保每个tensor都有_keras_history，以便keras能够全面地追踪并管理整个神经网络。

这一点，在我们使用tf.keras时要格外注意。

同时，进一步，可以更深刻地理解tensor和variable的区别。tensor是layer的输入和输出，在计算时才有值，同时，还记录了自己从哪里来这样的信息（即tensor坐标）。variable就是比较简单了，就是简单的多维数组，并且有初始化值。

这么联系起来一想，对keras就有了更深刻的感知了。

继续往下读：

![2022-02-21-keras中Layer源码解读(下)-20220221181153](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181153.png)

这里是创建了Node，之前已经看过，在创建Node时，会把该node加到input layer的outbound_nodes数组中，同时加入该layer的inbound_nodes数组中。

![2022-02-21-keras中Layer源码解读(下)-20220221181209](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181209.png)

有点绕，需要慢下来，确保自己弄清楚了。

Node添加完之后，就是为output tensor添加_keras_history

![2022-02-21-keras中Layer源码解读(下)-20220221181223](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181223.png)


关键看下面的一行。

(self, len(self._inbound_nodes) - 1, i)

因为Node创建时，该layer的_inbound_nodes已经添加了一个Node，所以，直接使用该Node的index作为第二个坐标。

上面的_uses_learning_phase是keras中一个便利标志。可以让每个tensor针对训练阶段进行一些定制操作。K.in_train_phase就是配套的方法。这一点可以以后慢慢理解，现在知道有这么回事儿就行。

确定每个output tensor的 _uses_learning_phase时有三个考虑，一个就是该tensor自己的_uses_learning_phase属性，另一个就是产生该tensor的layer的_uses_learning_phase属性，另一个就是所有的input tensor的_uses_learning_phase属性。代码中已经明示了。

至此，我们就读完了__call__方法。撒花！


# 608-666
这段代码中的几个方法，之前已经查看过：

compute_output_shape

compute_mask

build

如果没啥印象，请自己打开源码看看即可。

# 667-698

![2022-02-21-keras中Layer源码解读(下)-20220221181308](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181308.png)

这个方法是从layer的inbound_node中取信息的。

如果留心一下，发现只是从inbound_nodes数组中取，没有用outbound。

# 700-966
这里的方法都是对_get_node_attribute_at_index的运用。还是非常的简单直接的。这里展示其中一个方法：

![2022-02-21-keras中Layer源码解读(下)-20220221181332](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181332.png)

其他都是类似。

# 965-991
这段处理的是layer 的metrics。先看一下代码：
![2022-02-21-keras中Layer源码解读(下)-20220221181406](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181406.png)

这段代码主要是查询metric和添加metric。 添加metric的代码中涉及的几个方法都在base layer中。我们一并列出来：
![2022-02-21-keras中Layer源码解读(下)-20220221181421](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181421.png)

这是查询要添加的metric是否已经存在，防止重复添加同一个metric。

![2022-02-21-keras中Layer源码解读(下)-20220221181433](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181433.png)

当要添加的value没有_metric_obj属性时，默认生成的是Mean metric。并对value进行计算。生成的_metric_obj会添加到layer的_metrics属性中。

这里就体现了动态类型语言的灵活。当value是Metric对象时，将它的_metric_obj属性添加到metrics中。当value是要计算的值时，生成Mean Metric对象，对value执行计算后返回_metric_obj属性。

好奇的我，找到metric 的源码，看看_metric_obj是什么。

![2022-02-21-keras中Layer源码解读(下)-20220221181449](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181449.png)


可以看到_metric_obj就是 Metric的对象。

为啥要这么绕呢？

仔细看，这个属性是在call方法中添加的。当call的时候，说明是用了这个metric。这个时候，它内部的状态已经更新过至少一次了。所以，直接将这个对象加入metrics，才能保持状态的连续性。

设计的还是很巧妙。

到目前为止，我们还没有深入探究metric。但是既然读到这里，有必要进行一个简单的学习。于是，查找了一下资料。

![2022-02-21-keras中Layer源码解读(下)-20220221181509](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181509.png)

首先看到Metric也是一个Layer，我们学习到的layer的知识，对metric都管用。 然后，我查到另一些资料：

![2022-02-21-keras中Layer源码解读(下)-20220221181523](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181523.png)

这两段话很精准地说明了什么是metric。

今天先到这里。下一篇预计可以读完Layer所有源码。


今天继续Layer源码的学习。


# 992-1026

![2022-02-21-keras中Layer源码解读(下)-20220221181608](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-02-21-keras中Layer源码解读(下)-20220221181608.png)

这个添加loss的方法本身比较简单。但是，我进行了一个思考。

一般来说，我们对Model有loss比较熟悉。可是作者却在最基本的Layer中就实现了loss。大概是想说明，每一个Layer，有输入，有输出，输出都可能有一个标准答案，这个layer可以对照这个标准答案进行学习。

从整个神经网络网络来看，是很多的Layer，layer之间形成了复杂的连接，最后有一个output，这个最终的output一般是有label，即标准答案。整个Model是对照这个标准答案来学习。

layer可以有自己的标准答案。感觉起来，就是既要结果，也要过程。整体答案要对，每个layer的输出也要尽可能对。

这是一个理解layer也有自己的loss的角度。还可以从另外一个角度来理解。

即Model中每个layer的输出都是Model的输出，这么多输出，都可以有自己的label，Model都可以进行学习。

仔细体会这两种角度，前面的角度有点像联邦制，后面的角度有点像集中制。

# 1027-1087
共三个方法：

add_update

get_loss_for

get_update_for

这里面的add_update和add_loss很相似。另外两个方法看名字就知道了。

# 1088-1261
这部分都是一些getter和setter，还有一些小方法，之前都曾解读过。

# 1262-1425
这部分就是两个类：

InputSpec和Node

之前也都解读过了。

# 1426-1446
```python
def _collect_previous_mask(input_tensors):
    """Retrieves the output mask(s) of the previous node.

    # Arguments
        input_tensors: A tensor or list of tensors.

    # Returns
        A mask tensor or list of mask tensors.
    """
    input_tensors = to_list(input_tensors)
    masks = []
    for x in input_tensors:
        if hasattr(x, '_keras_history'):
            inbound_layer, node_index, tensor_index = x._keras_history
            node = inbound_layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        else:
            masks.append(None)
    return unpack_singleton(masks)
```
这个方法之前也解读过。这里再次贴出来，是想进一步加深tensor坐标的概念。mask正是通过tensor的坐标取出来的。

# 1447-1475
两个简单的小方法。一看就知道了。

# 总结
没想到1000多行的代码竟然写了10篇文章。之所以如此大费笔墨。实在是因为这个Layer是keras的奠基之类。后面的Model，Loss，Metric，Optimizer等概念都是在Layer的基础上的。

可以说，搞清楚Layer，就学会了keras的百分之八十。
