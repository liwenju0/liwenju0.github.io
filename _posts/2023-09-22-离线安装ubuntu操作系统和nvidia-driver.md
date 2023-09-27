---
layout: post
title:  "离线安装ubuntu操作系统和nvidia-driver"
date:   2023-09-22 11:20:08 +0800
category: "计算机基础"
published: true
---

 记录一下安装过程中遇到的问题。

<!--more-->
## 一、安装ubuntu的问题
安装的是 20.04 版本的 ubuntu 系统。机子上本来安装有 windows 系统，这次安装 ubuntu 是把机器装成双系统。

### 1.系统无法启动，黑屏
安装正常安装下来，安装是成功了，但是系统无法启动。症状就是输入用户名密码后黑屏，不再有反映，连光标闪烁都没有。

后来找到这个教程：
https://zhuanlan.zhihu.com/p/617640635

按照这个里面的教程，终于安装成功，核心区别就是

![2023-09-22-17-52-53](https://raw.githubusercontent.com/liwenju0/blog_pictures/master/pics/2023-09-22-17-52-53.png)

这一步中选择 **其他选项**

然后按照上面教程，依次建立几个分区，正确设置挂载目录。
这次安装后，系统就可以启动了。

### 2.离线安装 gcc 和 make

之所以要安装这个，是因为安装 nvidia driver 需要有 gcc。
所幸，在安装光盘的 Pool 目录下，下面涉及到的 deb 包大都可以找到。
参考的是这个教程：
https://blog.csdn.net/weixin_42432439/article/details/108777302

- sudo dpkg -i libc6_2.31-0ubuntu9.1_amd64.deb
- sudo dpkg -i manpages-dev_5.05-1_all.deb
- sudo dpkg -i binutils-common_2.34-6ubuntu1_amd64.deb
- sudo dpkg -i linux-libc-dev_5.4.0-48.52_amd64.deb
- sudo dpkg -i libctf-nobfd0_2.34-6ubuntu1_amd64.deb
- sudo dpkg -i libgomp1_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i libquadmath0_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i libmpc3_1.1.0-1_amd64.deb
- sudo dpkg -i libatomic1_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i libubsan1_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i libcrypt-dev_4.4.10-10ubuntu4_amd64.deb
- sudo dpkg -i libisl22_0.22.1-1_amd64.deb
- sudo dpkg -i libbinutils_2.34-6ubuntu1_amd64.deb
- sudo dpkg -i libc-dev-bin_2.31-0ubuntu9.1_amd64.deb
- sudo dpkg -i libcc1-0_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i liblsan0_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i libitm1_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i gcc-9-base_9.3.0-10ubuntu2_amd64.deb
- sudo dpkg -i libtsan0_10-20200411-0ubuntu1_amd64.deb
- sudo dpkg -i libctf0_2.34-6ubuntu1_amd64.deb
- sudo dpkg -i libasan5_9.3.0-10ubuntu2_amd64.deb
- sudo dpkg -i cpp-9_9.3.0-10ubuntu2_amd64.deb
- sudo dpkg -i libc6-dev_2.31-0ubuntu9.1_amd64.deb
- sudo dpkg -i binutils-x86-64-linux-gnu_2.34-6ubuntu1_amd64.deb
- sudo dpkg -i binutils_2.34-6ubuntu1_amd64.deb
- sudo dpkg -i libgcc-9-dev_9.3.0-10ubuntu2_amd64.deb
- sudo dpkg -i cpp_9.3.0-1ubuntu2_amd64.deb
- sudo dpkg -i gcc-9_9.3.0-10ubuntu2_amd64.deb
- sudo dpkg -i gcc_9.3.0-1ubuntu2_amd64.deb

**严格按照这个顺序进行安装**，期间有几个安装失败，就直接跳过了，事实证明，也可以成功。

安装 make 比较简单，就是一个 deb 包，在 Pool 中搜索到后直接安装即可。


## 二、安装 nvidia driver 的问题
在网上找到这个教程：https://blog.csdn.net/ChaoFeiLi/article/details/110945692

参考下来，有一点不同，就是 lightdm 我的系统上就没有。所以步骤有点不一样。我把使用到的步骤罗列如下：

### 1.禁用 nouveau驱动

lsmod | grep nouveau # 查看有没有输出，如果有信息输出，则需要禁掉

sudo gedit /etc/modprobe.d/blacklist.conf

在blacklist.conf的最后添加下面几行：

blacklist nouveau

options nouveau modeset=0

保存，关闭


更新

sudo update-initramfs -u

重启

lsmod | grep nouveau # 查看有没有输出，如果没有任何信息输出，则说明ok

Note：我实际执行时，这里blacklist 实际上还添加了几项。忘记是哪几个了，回头再来加上。

### 2、卸载 nvidia 原有驱动

sudo apt-get remove nvidia-*  

### 3、安装驱动

sudo sh NVIDIA-Linux-x86_64-440.31.run --no-opengl-files –no-x-check –no-nouveau-check

–no-opengl-files 只安装驱动文件，不安装OpenGL文件。这个参数最重要

–no-x-check 安装驱动时不检查X服务

–no-nouveau-check 安装驱动时不检查nouveau

后面两个参数可不加。

如果在装的过程中出现以下信息，请选择：

The distribution-provided pre-install script failed! Are you sure you want to continue?
选择 yes 继续。

Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later?
选择 No 继续。

Nvidia’s 32-bit compatibility libraries?
选择 No 继续。

Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up.
选择 no 继续

### 4、遇到的问题

安装好后，nvidia-smi 出现问题：

NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running

这个问题，网上资料很多，最终解决问题的是关闭系统的 secure boot

操作步骤是进入 bios，选择 security 选项卡，关闭 secure boot 即可。

看似简单，中间走了不少的弯路。














