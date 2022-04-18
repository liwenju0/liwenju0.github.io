---
layout: post
title:  "苏建林多标签softmax分类loss源码解析"
date:   2022-04-18 09:20:08 +0800
category: "AI"
published: true
---
看了一下苏神这篇博客:[将“softmax+交叉熵”推广到多标签分类问题](https://kexue.fm/archives/7359)。从单标签分类很自然地顺推到多标签。下面记录阅读其loss实现源码理解，以备忘查。

loss公式：

![2022-04-18-苏建林多标签softmax分类loss源码解析-20220418141355](https://cdn.jsdelivr.net/gh/liwenju0/blog_pictures@main/pics/2022-04-18-苏建林多标签softmax分类loss源码解析-20220418141355.png)


源码和解析：
```python
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    # 将正标签乘1，负标签乘-1
    y_pred = (1 - 2 * y_true) * y_pred
    # 将正标签的预测值设为无穷小
    y_pred_neg = y_pred - y_true * 1e12
    # 将负标签的预测值设为无穷小
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    # 预测值后面添加一个0， 为了上面公式中log里面的1
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    # 计算两个loss，和上面的公式对应
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss
```