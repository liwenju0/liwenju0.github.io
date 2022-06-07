---
layout: post
title:  "pytorch中的lr scheduler"
date:   2022-06-07 09:20:08 +0800
category: "AI"
published: true
---

总结一下pytorch中lr scheduler的核心逻辑。

<!--more-->
所有的lr scheduler 都是继承的_LRScheduler,查看它的源码，可以发现，核心逻辑在get_lr这个方法。
因此，找到各个lr scheduler的这个方法，就明白了核心逻辑。

# LambdaLR
```python

return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
```

# MultiplicativeLR
```python
if self.last_epoch > 0:
    return [group['lr'] * lmbda(self.last_epoch)
                    for lmbda, group in zip(self.lr_lambdas, self.optimizer.param_groups)]
else:
    return [group['lr'] for group in self.optimizer.param_groups]
```

上面两个scheduler差不多。

# StepLR
```python
if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
        return [group['lr'] for group in self.optimizer.param_groups]
return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
```
last_epoch变量是内部维护的一个变量，每次调用step会加1。所以和epoch并不必然相关。
只是大家平时使用时，在每个epoch结束时调用step方法，才让last_epoch和epoch同步起来了。

这里的step_size并不是训练模型时的step，而是内部用来决定是否更新学习率的一个控制变量。

这个scheduler本意就不是每个epoch结束调用的，而是为了每过多少个step衰减一次学习率设计的。


# MultiStepLR
```python
if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
```

这里的milestones中可以多次重复一个milestone。self.milestones[self.last_epoch]就是计算出来的次数。

# ConstantLR
```python
if self.last_epoch == 0:
    return [group['lr'] * self.factor for group in self.optimizer.param_groups]

if (self.last_epoch > self.total_iters or
        (self.last_epoch != self.total_iters)):
    return [group['lr'] for group in self.optimizer.param_groups]

if (self.last_epoch == self.total_iters):
    return [group['lr'] * (1.0 / self.factor) for group in self.optimizer.param_groups]
```
达到一定的epoch后，直接将学习率缩放一次，就不再变化了。

# LinearLR
```python
if self.last_epoch == 0:
    return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

if (self.last_epoch > self.total_iters):
    return [group['lr'] for group in self.optimizer.param_groups]

return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
        (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
        for group in self.optimizer.param_groups]
```
默认实现是学习率递增，且递增的幅度线性减小。并非学习率线性减小。


# ExponentialLR
```python
 if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
```
按照gamma不断衰减。

# SequentialLR
```python
idx = bisect_right(self._milestones, self.last_epoch)
if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
    self._schedulers[idx].step(0)
else:
    self._schedulers[idx].step()
self._last_lr = self._schedulers[idx].get_last_lr()
```
上面的代码是在step方法中，为啥没有在get_lr呢？因为这个schduler就是按照milestones来使用其他的schduler。

# CosineAnnealingLR

```python
if self.last_epoch == 0:
    return [group['lr'] for group in self.optimizer.param_groups]
elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
    return [group['lr'] + (base_lr - self.eta_min) *
            (1 - math.cos(math.pi / self.T_max)) / 2
            for base_lr, group in
            zip(self.base_lrs, self.optimizer.param_groups)]
return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
        (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
        (group['lr'] - self.eta_min) + self.eta_min
        for group in self.optimizer.param_groups]
```
公式比较复杂，看一下例子
```python
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=1.)
steps = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

for epoch in range(5):
    for idx in range(steps):
        scheduler.step()
        print(scheduler.get_lr())
    
    print('Reset scheduler')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
```
作用就是每个epoch中学习率逐渐减小，直到一个最小值。新的epoch开始时，又重新开始上一个epoch的学习率衰减。

# CylicLR
```python
cycle = math.floor(1 + self.last_epoch / self.total_size)
x = 1. + self.last_epoch / self.total_size - cycle
if x <= self.step_ratio:
    scale_factor = x / self.step_ratio
else:
    scale_factor = (x - 1) / (self.step_ratio - 1)

lrs = []
for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
    base_height = (max_lr - base_lr) * scale_factor
    if self.scale_mode == 'cycle':
        lr = base_lr + base_height * self.scale_fn(cycle)
    else:
        lr = base_lr + base_height * self.scale_fn(self.last_epoch)
    lrs.append(lr)
```
学习率在最大最小之间循环，并且更新的频率保持不变。



