---
layout: post
title:  "triton server 模型的加解密"
date:   2024-05-09 11:20:08 +0800
category: "人工智能"
published: true
---

Triton server如何实现模型的加解密，这对私有部署模型，担心模型权重泄露的场景很有价值。本文记录笔者实现这一目标的过程。
先定义一个最简单的加解密方案。加密就是在onnx模型文件的开头添加一个字节00000000，解密就是去掉这个字节。
因为只是想跑通流程，这个加解密方案已经可以满足目的了。


<!--more-->

## 基本资料
官方资料上，repository agent中提到了模型的加解密。地址是：
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/repository_agents.html

另外，在github中的issue中，也有人探讨了这个话题。

https://github.com/triton-inference-server/server/issues/1777

但最终也没有找到一个完整的样例。

只能自己探索了。

## 编译好checksum repository agent

下载好官方的repo。
https://github.com/triton-inference-server/checksum_repository_agent

下面的安装过程，都是在我机器上报错后，根据报错信息安装的，如果你没看到相关的信息，说明已经安装好了。

### 安装编译工具
sudo apt update
sudo apt install build-essential

### 安装rapidjson

git clone https://github.com/Tencent/rapidjson.git

cd rapidjson
mkdir build
cmake ..

sudo make install 
这里使用sudo是为了防止没有目录权限。

### 安装openssl
sudo apt-get install libssl-dev
sudo find / -name md5.h

### 编译成功

rm -rf build&&mkdir build&&cd build&&cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..&&make install

一串命令下来，应该在项目的build目录下发现：
libtritonrepoagent_checksum.so
这就是最终的成品。


## 修改成加解密

找遍文档，在tritonrepoagent.h中发现了一段跟加解密相关的说明。

/// Any modification to the model's repository must be made when 'action_type'

/// is TRITONREPOAGENT_ACTION_LOAD.

/// To modify the model's repository the agent must either acquire a mutable

/// location via TRITONREPOAGENT_ModelRepositoryLocationAcquire

/// or its own managed location, report the location to Triton via

/// TRITONREPOAGENT_ModelRepositoryUpdate, and then return

/// success (nullptr). If the agent does not need to make any changes

/// to the model repository it should not call

/// TRITONREPOAGENT_ModelRepositoryUpdate and then return success.

/// To indicate that a model load should fail return a non-success status.

意思就是通过TRITONREPOAGENT_ModelRepositoryLocationAcquire获得一个临时目录，
将解密后的模型放在这里。
然后通过TRITONREPOAGENT_ModelRepositoryUpdate告诉Triton，来加载这里的模型。
最后返回nullptr表示成功。

然后查看TRITONREPOAGENT_ModelRepositoryLocationAcquire的说明。

/// Acquire a location where the agent can produce a new version of

/// the model repository files. This is a convenience method to create

/// a temporary directory for the agent. The agent is responsible for

/// calling TRITONREPOAGENT_ModelRepositoryLocationDelete in

/// TRITONREPOAGENT_ModelFinalize to delete the location. Initially the

/// acquired location is empty. The 'location' communicated depends on

/// the requested 'artifact_type'.
///
///   TRITONREPOAGENT_ARTIFACT_FILESYSTEM: The location is a directory

///     on the local filesystem. 'location' returns the full path to

///     an empty directory that the agent should populate with the

///     model's artifacts. The returned location string is owned by

///     Triton, not the agent, and so should not be modified or freed.

核心意思， location二级指针用来接收Triton返回的新目录，这个目录属于Triton，只能往里面放新的模型。
按照这些说明，写出了如下代码：
```c++
TRITONSERVER_Error*
TRITONREPOAGENT_ModelAction(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const TRITONREPOAGENT_ActionType action_type)
{
  const char* location_cstr;
  TRITONREPOAGENT_ArtifactType artifact_type;
  RETURN_IF_ERROR(TRITONREPOAGENT_ModelRepositoryLocation(
      agent, model, &artifact_type, &location_cstr));
  if (artifact_type != TRITONREPOAGENT_ARTIFACT_FILESYSTEM) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Unsupported filesystem: ") +
         std::to_string(artifact_type))
            .c_str());
  }

  if (action_type == TRITONREPOAGENT_ACTION_LOAD) {
    const char* decrypt_location;
    RETURN_IF_ERROR(TRITONREPOAGENT_ModelRepositoryLocationAcquire(
        agent, model, artifact_type, &decrypt_location));

    RETURN_IF_ERROR(copyDirectoryAndDecrypt(location_cstr, decrypt_location));

    RETURN_IF_ERROR(TRITONREPOAGENT_ModelRepositoryUpdate(
        agent, model, artifact_type, decrypt_location));
  }

  // if(action_type == TRITONREPOAGENT_ACTION_LOAD_COMPLETE){
  //    std::cout << "TRITONREPOAGENT_ACTION_LOAD_COMPLETE!!!!!!!!!!" << std::endl;
  // }

  return nullptr;  // success
}
```
核心逻辑在copyDirectoryAndDecrypt方法中，它的主要作用就是将现在model_repo中的文件，复制到申请到的临时目录中。
如果文件是onnx的模型文件，执行解密后再存在临时目录里。

解密函数的内容：
```c++
TRITONSERVER_Error*
decrypt(const std::string& inputPath, const std::string& outputPath)
{
  namespace fs = std::filesystem;
  // 检查输入文件是否存在
  if (!fs::exists(inputPath)) {
    std::cerr << "Error: 输入文件 '" << inputPath << "' 不存在!" << std::endl;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND, "Error: 输入文件  不存在!");
  }

  // 检查输出文件是否存在，如果存在则删除
  if (fs::exists(outputPath)) {
    std::cerr << "临时文件存在，先删除：" << outputPath << std::endl;

    fs::remove(outputPath);
  }

  fs::path output_dir = fs::path(outputPath).parent_path();  // 获取父目录路径

  // 检查目录是否存在，如果不存在则创建
  if (!fs::exists(output_dir)) {
    if (!fs::create_directories(output_dir)) {
      std::cerr << "无法创建目录：" << output_dir << std::endl;
    }
    std::cout << "已创建目录：" << output_dir << std::endl;
  }

  // 以二进制方式打开输入和输出文件
  std::ifstream inputFile(inputPath, std::ios::binary);
  std::ofstream outputFile(outputPath, std::ios::binary);

  // 检查文件流是否成功打开
  if (!inputFile.is_open() || !outputFile.is_open()) {
    std::cerr << "Error: 文件打开失败!inputPath: " << inputFile.is_open()
              << inputPath << "outputPath: " << outputFile.is_open()
              << outputPath << std::endl;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Error: 文件打开失败!");
  }

  // 将输入文件从第二个字节开始的内容写入到输出文件
  inputFile.seekg(1);  // 跳过第一个字节
  outputFile << inputFile.rdbuf();

  // 关闭文件流
  inputFile.close();
  outputFile.close();

  // 输出成功信息
  std::cout << "解密成功，输出文件已生成: " << outputPath << std::endl;

  return nullptr;
}
```

比较顺利，一次成功了！
## 删除解密的模型
在上面解密模型时，我们还是放到了一个临时目录下，如果别人拿到这里面的模型，还是会导致加密失效。
所以我要在合适的时机，将这个解密后的模型删除。

看上面的代码，下面有三行注释的内容。这是我打算在TRITONREPOAGENT_ACTION_LOAD_COMPLETE阶段，将解密文件删除。
但是发现，根本没调用到！
查了下资料，这确实是个bug：https://github.com/triton-inference-server/server/issues/6359
前几天刚刚修复，得等新的容器版本更新了。


## 总结
在本文中，我探索了如何在Triton Inference Server中实现模型的加解密。我首先定义了一个简单的加解密方案，即在ONNX模型文件的开头添加一个字节作为加密，解密时去除这个字节。为了实现这一过程，我参考了官方文档和GitHub上的讨论，但未找到完整的样例。因此，我自行编译了checksum repository agent，并根据tritonrepoagent.h中的说明，通过TRITONREPOAGENT_ModelRepositoryLocationAcquire获得临时目录，将解密后的模型存放于此，并通过TRITONREPOAGENT_ModelRepositoryUpdate告知Triton加载此模型。然而，我遇到了一个bug，即在模型加载成功后，TRITONREPOAGENT_ACTION_LOAD_COMPLETE事件并未被调用，导致无法在合适的时机删除解密后的模型。这个问题在GitHub上已被记录并修复，我需要等待新的容器版本更新。通过这次实践，我深入理解了Triton server的模型加解密流程，并成功实现了模型的加密保护。（Kimi总结）






