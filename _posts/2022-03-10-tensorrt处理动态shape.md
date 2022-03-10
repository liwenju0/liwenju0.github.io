---
layout: post
title:  "tensorrt处理动态shape"
date:   2022-03-10 10:20:08 +0800
category: "AI"
published: true
---

> 使用tensorrt时如何处理动态shape


<!--more-->

### 1、onnx的动态输出
若要使用tensorrt的动态shape支持，那么在导出onnx时就应该指定动态shape的dimension。详见下面的代码中的dynamic_axes。

```python
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

如果不在输出onnx时指定动态shape的维度，那么在使用tensorrt的动态shape支持时，可能会报错，注意，这里是可能会报错。

因为如果某个layer中使用的hard code的shape参数，导出成onnx时没有声明动态shape，则tensorrt在将onnx 解析成engine时，会出现错误。
详见这个回答：[https://forums.developer.nvidia.com/t/working-with-dynamic-shape-example/112738/3](https://forums.developer.nvidia.com/t/working-with-dynamic-shape-example/112738/3)。

### 2、tensorrt对动态shape的支持演进

tensorrt发展很快。早期的版本中，tensorrt有两种运行模式，explicit batch 和implicit batch。
implicit batch模式下，默认每个tensor都有第0维都是batch，其他的维度是完全一样并且是固定不变的。

explicit batch 模式下，tensor所有的维度都需要明确指定并且是动态的。这是tensorrt为了支持动态shape开发的模式。

当前，implicit batch已经处于过期状态。官方只推荐explicit batch模式。不过，由于历史遗留原因，这个名字中虽然有batch，但实际上所有的维度都是可变的，需要留心。

### 3、支持动态shape引入的概念
onnx导出时虽然指定了动态的dimension。但是，使用tensorrt转化成engine时，最好还是给一些提示，指定这些维度的取值范围，这可以帮助tensorrt生成性能更好的engine。这就是optimization profile的作用。
如下：
> An optimization profile describes a range of dimensions for each network input and the dimensions that the auto-tuner will use for optimization. 

举例来讲，一个optimization profile可能是如下所示：

最小的shape：   [3,100,200]

最大的shape：  [3,200,300]

优化用shape：   [3,150,250]

这对应到trtexec中的--minShapes 、 --maxShapes 和 --optShapes三个参数。

深入一点，一个engine可以支持多个optimization profile。在实际使用时需要指定使用哪个profile。profile是按照0，1，2...的顺序进行编码。


### 4、使用示例

首先，导出成onnx时使用dynamic。
```python
def export_onnx(model,image_shape,onnx_path, batch_size=1):
    x,y=image_shape
    img = torch.zeros((batch_size, 3, x, y))
    dynamic_onnx=True
    if dynamic_onnx:
        dynamic_ax = {'input_1' : {2 : 'image_height',3:'image_wdith'},   
                                'output_1' : {2 : 'image_height',3:'image_wdith'}}
        torch.onnx.export(model, (img), onnx_path, 
           input_names=["input_1"], output_names=["output_1"], verbose=False, opset_version=11,dynamic_axes=dynamic_ax)
    else:
        torch.onnx.export(model, (img), onnx_path, 
           input_names=["input_1"], output_names=["output_1"], verbose=False, opset_version=11
    )
```

第二，构建engine时添加profile

```python
def build_engine(onnx_path, using_half,engine_file,dynamic_input=True):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(1)
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        ##增加部分
        if dynamic_input:
            profile = builder.create_optimization_profile();
            profile.set_shape("input_1", (1,3,512,512), (1,3,1600,1600), (1,3,1024,1024)) 
            config.add_optimization_profile(profile)
        #加上一个sigmoid层
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)
        sigmoid_layer=network.add_activation(previous_output,trt.ActivationType.SIGMOID)
        network.mark_output(sigmoid_layer.get_output(0))
        return builder.build_engine(network, config) 
```

第三，使用时指定实际的shape和使用的profile。

```python
def profile_trt(engine, imagepath,batch_size):
    assert(engine is not None)  
    
    input_image,input_shape=preprocess_image(imagepath)

    segment_inputs, segment_outputs, segment_bindings = allocate_buffers(engine, True,input_shape)
    
    stream = cuda.Stream()    
    with engine.create_execution_context() as context:
        context.active_optimization_profile = 0#增加部分
        origin_inputshape=context.get_binding_shape(0)
        #增加部分
        if (origin_inputshape[-1]==-1):
            origin_inputshape[-2],origin_inputshape[-1]=(input_shape)
            context.set_binding_shape(0,(origin_inputshape))
        input_img_array = np.array([input_image] * batch_size)
        img = torch.from_numpy(input_img_array).float().numpy()
        segment_inputs[0].host = img
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in segment_inputs]#Copy from the Python buffer src to the device pointer dest (an int or a DeviceAllocation) asynchronously,
        stream.synchronize()#Wait for all activity on this stream to cease, then return.
       
        context.execute_async(bindings=segment_bindings, stream_handle=stream.handle)#Asynchronously execute inference on a batch. 
        stream.synchronize(）
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in segment_outputs]#Copy from the device pointer src (an int or a DeviceAllocation) to the Python buffer dest asynchronously
        stream.synchronize()
        results = np.array(segment_outputs[0].host).reshape(batch_size, input_shape[0],input_shape[1])    
    return results.transpose(1,2,0)
```


Refs:

[https://zhuanlan.zhihu.com/p/299845547](https://zhuanlan.zhihu.com/p/299845547
)

[https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles)








