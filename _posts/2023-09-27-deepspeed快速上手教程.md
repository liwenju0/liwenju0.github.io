---
layout: post
title:  "deepspeed快速上手教程"
date:   2023-09-27 11:20:08 +0800
category: "AI"
published: true
---

记录一下快速使用deepspeed的基本操作。

<!--more-->

deepspeed执行训练的核心抽象是DeepSpeedEngine，它负责管理整个训练过程，包括数据读取、模型训练、反向传播、优化器更新等。
Engine可以封装任意的torch.nn.Module，因此DeepSpeed可以支持任何PyTorch模型。

## 一、初始化
```python
model_engine, optimizer, dataloader, lr_schduler = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     training_data=training_data,
                                                     lr_scheduler=lr_scheduler,
                                                     model_parameters=params)
```
该操作可以完成的封装功能有：
- 分布式训练初始化
- 混合精度训练初始化
- 学习率调度器初始化
- 分布式数据加载器初始化

## 二、训练
使用三个基本的api进行模型的训练
```python
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

```
在这些操作背后，deepspeed会自动完成分布式训练需要的操作。
- 梯度平均
- 损失缩放
- 学习率更新

注意，这里的学习率更新是在每次```model_engine.step()```之后，由deepspeed自动完成的。如果学习率更新不是按照step，而是按照epoch，则不能将学习率调度器传入init函数，然后自己手动执行学习率调度器的更新。

## 三、保存模型

保存模型就是用两个简单的load和save的api，如下所示：
```python
#load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

#advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):

    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

    #save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)

```

可以看到，通过简单保存step的值，deepspeed可以实现从任意step恢复训练。

至于其他的，模型参数，优化器状态，学习率调度器状态，都由deepspeed自动保存和恢复了。不需要我们关心。

client_sd用来保存我们自己需要单独定制的参数状态，比如step。

**Note：** 所有的进程都需要调用这个保存checkpoint的操作，因为所有的进程都需要保存master的参数状态以及各自优化器和学习率调度器的状态。

## 四、deepspeed的配置
deepspeed的配置文件是json格式的，说明文档非常详尽。这一点真的非常赞！好的开源项目就应该这样。

## 五、多机多卡
deepspeed使用hostfile配置文件，完成多机多卡的分布式训练。该配置文件的格式如下：
```
worker-1 slots=4
worker-2 slots=4
```
worker-1和worker-2是机器名，需要在hosts中配置好，且通过ssh-copy-id确保所有机器之间能够免密访问。

slots是该机器上使用的卡数。

具体使用的代码如下：
```bash
#直接使用hostfile
deepspeed --hostfile=myhostfile <client_entry.py> <client args> \
  --deepspeed --deepspeed_config ds_config.json

#使用指定数量的节点
deepspeed --num_nodes=2 \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json

#排除特定的节点上的特定gpu
deepspeed --exclude="worker-2:0@worker-3:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json

#指定特定的节点上的特定gpu
deepspeed --include="worker-2:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json

```

## 六、多节点的环境变量配置
如果需要在多个节点上使用相同的环境变量。在home目录下创建.deepspeed_env文件，设置好环境变量的值，如：
```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```
那么，deepspeed可以确保每个进程都使用相同的环境变量配置。

