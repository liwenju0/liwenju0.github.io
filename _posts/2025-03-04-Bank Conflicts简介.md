---
layout: post
title:  "Bank Conflicts简介"
date:   2025-03-05 08:20:08 +0800
category: "AI"
published: true
---

什么是Bank Conflicts？如何解决这个问题？
<!--more-->

## 1、什么是Bank Conflicts？
在cuda里面，可以使用共享内存来加速计算。共享内存在**物理上**，被分成一个个大小相等的内存模块，每个模块叫做一个Bank。这些Bank之间是可以同时进行读写的。

如果**同时**有n个读写共享内存的请求，这n个请求访问的共享内存地址**恰好在不同的Bank上**，那么这n个请求可以同时进行，显然，这能够大大提升访存效率。

如果不巧，n个共享内存地址在一个Bank上，那么这n个读写请求只能顺序进行了，这就是Bank Conflicts。


为了规避Bank Conflicts，就是将多个线程访问的共享内存地址**错开**，最好是每个Bank上只有一个线程访问，自然就不会出现Bank Conflicts了。

为了达到这个目的，我们需要知道共享内存地址是如何映射到具体的Bank上的。

![20250304110327](https://raw.githubusercontent.com/liwenju0/blog_pictures/main/pics/20250304110327.png)

如上图所示，共享内存被划分成了32个Bank。正好和一个warp的线程数量相同。

**共享内存地址映射到Bank的方式是：连续的32个地址被映射到连续的不同的32个Bank上。以此类推。**


## 2、上手看看？

下面，我们首先写一个有Bank Conflicts的代码，然后看看如何解决这个问题。

```c
// ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bank_conflict_test

__global__ void bank_conflict_test(float *a, float *b, float *c, int N) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    
    // 加载数据到共享内存
    shared_data[tid] = b[tid];
    __syncthreads();
    
    // 产生bank conflict的访问模式
    // 假设warp大小为32，每个线程访问相隔32个元素的位置
    // 这样同一warp中的所有线程都会访问同一个bank
    int conflict_idx = (tid % 32) * 32 + (tid / 32);
    if (conflict_idx < 256) {
        a[tid] = shared_data[conflict_idx] + c[tid];
    } else {
        a[tid] = b[tid] + c[tid];
    }
}
```

![20250304131704](https://raw.githubusercontent.com/liwenju0/blog_pictures/main/pics/20250304131704.png)

可以看到这种情况下，确实有56次访问共享内存的bank conflict。

l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum 后面的ld代表读。

l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum 后面的st代表写。

那我们怎么消除呢？如下所示：

```c
// ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bank_conflict_test

__global__ void bank_conflict_test(float *a, float *b, float *c, int N) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    
    // 加载数据到共享内存
    shared_data[tid] = b[tid];
    __syncthreads();
    
    // 顺序访问，就不会产生bank conflict
    int conflict_idx = tid;
    if (conflict_idx < 256) {
        a[tid] = shared_data[conflict_idx] + c[tid];
    } else {
        a[tid] = b[tid] + c[tid];
    }
}
```

![20250304132720](https://raw.githubusercontent.com/liwenju0/blog_pictures/main/pics/20250304132720.png)

可以看到，bank conflict为 0 。





