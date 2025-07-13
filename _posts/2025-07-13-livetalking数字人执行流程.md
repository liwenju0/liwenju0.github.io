---
layout: post
title:  "livetalking数字人执行流程"
date:   2025-07-13 09:20:08 +0800
category: "前后端技术"
published: true
---

使用debug模式记录livetalking数字人执行流程，使用的模型是musetalk。

<!--more-->

## 1、加载模型

```python
def load_all_model():
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    vae = VAE(model_path = "./models/sd-vae-ft-mse/")
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path ="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe
```
主要是这几个模型：
- whipser 用来从音频中提取特征，作为musetalk的输入
- vae 用来将图片编码到潜在空间，作为musetalk的输入
- unet 就是所谓的musetalk模型，这是一个扩散模型，根据音频和图像潜在表示，生成对应音频的潜在表示，最后再通过vae恢复成原图

整体流程看似简单，但细节繁多，理解并优化这些细节需要大量投入，这也是作者的核心壁垒。
很多情况下，直接购买商业版反而更高效。通过这个项目，我也更深刻体会到开源项目商业化的本质。

pe是位置编码，用来将音频和图像潜在表示进行位置编码，作为musetalk的输入。

## 2、加载数字人形象
```python
def load_avatar(avatar_id):
    #self.video_path = '' #video_path
    #self.bbox_shift = opt.bbox_shift
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"
    # self.avatar_info = {
    #     "avatar_id":self.avatar_id,
    #     "video_path":self.video_path,
    #     "bbox_shift":self.bbox_shift   
    # }

    input_latent_list_cycle = torch.load(latents_out_path, weights_only=False)  #,weights_only=True
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle
```
这里只关注最后return的几个值：
- frame_list_cycle 数字人形象的原始图片列表，按照时间顺序从视频中抽取的原始帧的图片。
- mask_list_cycle，mask_coords_list_cycle数字人形象的下半个脸的mask图像列表和对应的坐标列表
如下图所示：
![00000000](https://raw.githubusercontent.com/liwenju0/blog_pictures/main/pics/00000000.png)

- coord_list_cycle、input_latent_list_cycle数字人下半个脸的精确坐标列表和对应vae嵌入列表

这里需要留心mask_coords_list_cycle和coord_list_cycle之间的关系。coord_list_cycle是精确的下半脸的图像坐标，用来输入musetalk模型做预测。
mask_coords_list_cycle是在coord_list_cycle基础上做了扩大的图像坐标，用来将预测图像贴回原图时做线性融合使用。
可参考如下代码：
```python
# simple_musetalk.py
def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s
```
这是从coord_list_cycle中的坐标扩展到mask_coords_list_cycle坐标的代码。

再来看看贴回的代码：

```python
#blending.py
def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    face_large[y-y_s:y1-y_s, x-x_s:x1-x_s]=face

    mask_image = cv2.cvtColor(mask_array,cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image/255).astype(np.float32)
    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large,body[y_s:y_e, x_s:x_e],mask_image,1-mask_image)

    return body
```
face_large是根据mask_coords_list_cycle中的坐标得来，然后
```python
face_large[y-y_s:y1-y_s, x-x_s:x1-x_s]=face 
```
这行代码是把预测的精确图像嵌入到这个里面。
最后，将face_large贴回到原图像。

## 3、回答内容送入队列msgqueue

在对话系统中，用户输入内容后，会调用大模型获得回复。这些回复的文本内容会被分块处理，然后放入TTS队列中进行语音合成。

### 文本分块逻辑

#### 实现代码
```python
def llm_response(message,nerfreal:BaseReal):
    start = time.perf_counter()
    from openai import OpenAI
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 填写DashScope SDK的base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': message}],
        stream=True,
        # 通过以下设置，在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True}
    )
    result=""
    first = True
    for chunk in completion:
        if len(chunk.choices)>0:
            #print(chunk.choices[0].delta.content)
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start}s")
                first = False
            msg = chunk.choices[0].delta.content
            lastpos=0
            #msglist = re.split('[,.!;:，。！?]',msg)
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；" :
                    result = result+msg[lastpos:i+1]
                    lastpos = i+1
                    if len(result)>10:
                        logger.info(result)
                        nerfreal.put_msg_txt(result)
                        result=""
            result = result+msg[lastpos:]
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    nerfreal.put_msg_txt(result)    
```

### 分块规则
- **分块触发条件**：`if len(result)>10:` - 当累积的文本长度超过10个字符时进行分块
- **分割标点符号**：`",.!;:，。！？：；"` - 遇到这些标点符号时进行分割
- **数据流向**：分块后的文本通过 `nerfreal.put_msg_txt(result)` 送入TTS队列

### TTS队列管理

### BaseTTS类结构
```python
class BaseTTS:
    def __init__(self, opt, parent:BaseReal):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg:str,eventpoint=None): 
        if len(msg)>0:
            self.msgqueue.put((msg,eventpoint))
```

### 队列操作
- **入队**：`put_msg_txt(msg, eventpoint)` - 将文本消息和事件点放入队列，这个事件点在下面再次提到eventpoint时，会转变成msgevent，请留心下。
- **清空队列**：`flush_talk()` - 清空msgqueue并暂停状态

### 关键音频参数

#### 参数定义
```python
self.fps = opt.fps # 20 ms per frame
self.sample_rate = 16000
self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
```

#### 参数解析
- **fps**：音频帧率，设置为每帧20ms，即每秒50帧（1000ms ÷ 20ms = 50）
- **sample_rate**：音频采样率，16000Hz，表示每秒采集16000个音频采样点
- **chunk**：每帧包含的采样点数量，计算公式为 `sample_rate ÷ fps = 16000 ÷ 50 = 320`

### 参数关系理解
- **帧（frame）**：时间上的片段单位，此处为20ms
- **采样率（sample_rate）**：每秒钟采集的音频点数
- **chunk**：每个时间帧（20ms）内包含的采样点数量

### 重要性说明
这三个参数对以下功能至关重要：
- 音频与视频的同步
- 后续送入musetalk进行预测
- TTS分帧处理

### 数据流与事件系统

#### 音频数据流
- `self.input_stream = BytesIO()` - 存储TTS后的音频波形数据流

#### 事件点机制
`put_msg_txt`方法中的eventpoint作为系统事件总线，内容示例：
```python
eventpoint={'status':'start','text':text,'msgevent':textevent}
```
这里的msgevent，就是上面put_msg_txt提到的eventpoint。这里相当于是有一个层级机制。


#### 事件点功能
- 可传递到前端网页
- 控制音频视频同步
- 实现字幕同步高亮
- 支持更复杂的控制功能

## 4、tts处理msgqueue

我们看BaseTTS中的如下代码：
```python
class BaseTTS:
    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
```
在这里，`render`函数会启动一个新线程，在线程中执行`process_tts`方法。`process_tts`方法通常由各个TTS具体子类实现，用于处理消息队列中的文本转语音任务。

线程的启动与停止依赖于`quit_event`，这是一个线程安全的事件对象，用于统一管理各线程的生命周期。其定义如下：
```python
quit_event = threading.Event()
quit_event.set()
```
下面我们结合类EdgeTTS看看tts的具体处理：
```python
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        text,textevent = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint=None
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text,'msgevent':textevent}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')
```
`txt_to_audio`的整体处理逻辑是，先通过self.__main生成音频wavform数据，存入self.input_stream中。
然后，在`__create_bytes_stream`中，使用soundfile这个库读取self.input_stream中的数据，得到采样数据和采样率，采样数据如果是多声道，则只取第一个声道。
最后，如果采样率不是16000Hz，则使用resampy这个库进行重采样，使其满足16000Hz的采样率。
最后，返回采样数据。
这样，一段文本，就变成一个音频数据流。也就是一个float32的一维数组。

之后，就按照320个一个chunk，送入parent.put_audio_frame中。这个最后是调用的BaseASR中的put_audio_frame方法。

```python
def put_audio_frame(self,audio_chunk,eventpoint=None): #16khz 20ms pcm
    self.queue.put((audio_chunk,eventpoint))
```
这里，self.queue就是BaseASR中的queue，也就是音频数据流。
这样，音频数据流就送入到BaseASR中，等待后续处理。

至此，我们简短总结下处理流程。llm生成的回复，经过文本分块，送入TTS队列，TTS队列中的文本，经过tts处理，生成音频数据流，送入BaseASR中。

## 5、asr处理queue
进入BaseASR中的queue的音频数据流，是怎么被使用的呢？
按照设计，应该是传递给whisper模型，提取特征，然后送入musetalk模型，得到口型，然后将口型贴到对应的视频帧，最后，将视频帧和语音一起发送给前端。

该项目到这里涉及大量的异步线程，导致追踪执行过程比较难。我只能先按照我自己的思路记录下来处理过程，最后看看能不能按照线程逻辑串起来吧。

### 提取特征
这一步是在MuseAsr中的run_step中完成的。
```python
def run_step(self):
    ############################################## extract audio feature ##############################################
    start_time = time.time()
    for _ in range(self.batch_size*2):
        audio_frame,type,eventpoint = self.get_audio_frame()
        self.frames.append(audio_frame)
        self.output_queue.put((audio_frame,type,eventpoint))
    
    if len(self.frames) <= self.stride_left_size + self.stride_right_size:
        return
    
    inputs = np.concatenate(self.frames) # [N * chunk]
    whisper_feature = self.audio_processor.audio2feat(inputs)
    whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=self.fps/2,batch_size=self.batch_size,start=self.stride_left_size/2 )
    self.feat_queue.put(whisper_chunks)
    self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
```
这里关注几个点：
1、因为音频是20ms一帧，视频是40ms一帧，所以，这里需要提取batch_size*2个音频帧，然后送入whisper模型，提取特征。才能保证视频和音频同步。
2、我debug时设置的batch size是64，结果frames是148，多了20帧，这20帧在开始的warm up阶段被填充为全0。正好self.stride_left_size + self.stride_right_size相加为20。然后run_step结束后，还会继续保留20帧，如下代码所示
```python
self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
```
3、这里同步将音频帧送入output_queue中，output_queue是用来给webrtc使用的。看起来和BaseASR中的queue是同样的内容，但是从数据结构上看， output_queue多进程安全的。
```python
 self.queue = Queue()
self.output_queue = mp.Queue()
```
下面看看whisper是如何处理self.frames的。
经过```whisper_feature = self.audio_processor.audio2feat(inputs)```得到的`whisper_feature`的shape是
`(148, 5, 384)`。
这里面的音频变换过程，我点进去看了看，比较复杂，汉宁窗、短时傅里叶变换等等，这里先不深究。总之得到的结果第一维度是和self.frames的数量对齐的。知道这一点就不妨碍继续下去。
随后就是`feature2chunks`方法，这一点需要特别留心，musetalk和musetalk1.5中关于这个方法是不同的，一不小心就会搞错。
这个方法是和具体的唇形对齐模型高度相关的。因为需要将音频特征的shape整理成唇形对齐模型需要的。
总体的逻辑，可见：
```python
def feature2chunks(self,feature_array,fps,batch_size,audio_feat_length = [2,2],start=0):
    whisper_chunks = []
    whisper_idx_multiplier = 50./fps 
    i = 0
    #print(f"video in {fps} FPS, audio idx in 50FPS")
    for _ in range(batch_size):
        # start_idx = int(i * whisper_idx_multiplier)
        # if start_idx>=len(feature_array):
        #     break
        selected_feature,selected_idx = self.get_sliced_feature(feature_array= feature_array,vid_idx = i+start,audio_feat_length=audio_feat_length,fps=fps)
        #print(f"i:{i},selected_idx {selected_idx}")
        whisper_chunks.append(selected_feature)
        i += 1
        

    return whisper_chunks

def get_sliced_feature(self,
                           feature_array, 
                           vid_idx, 
                           audio_feat_length=[2,2],
                           fps=25):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return: 
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []
        
        center_idx = int(vid_idx*50/fps) 
        left_idx = center_idx-audio_feat_length[0]*2
        right_idx = center_idx + (audio_feat_length[1]+1)*2
        
        for idx in range(left_idx,right_idx):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)# 50*384
        return selected_feature,selected_idx

```
claude给出的简单解说：
```
1、feature2chunks函数：

循环batch_size次，每次处理一个视频帧
对于每个视频帧位置i，调用get_sliced_feature提取对应的音频特征片段
将所有提取的特征片段收集到whisper_chunks列表中

2、get_sliced_feature函数：

计算中心位置：center_idx = int(vid_idx*50/fps)，这是关键的帧率转换

由于音频特征是50fps，视频是25fps，所以视频第i帧对应音频的第i*2个位置


确定提取范围：以中心位置为基准，向左右扩展

左边界：center_idx - audio_feat_length[0]*2
右边界：center_idx + (audio_feat_length[1]+1)*2


提取特征：在这个范围内逐个取出音频特征，并处理边界情况（不超出数组范围）
最后将提取的特征拼接成固定形状(-1, 384)

简单理解
假设audio_feat_length=[2,2]，那么每个视频帧会提取对应的10个音频特征帧（左4个+右6个），这样可以为每个视频帧提供足够的音频上下文信息。

需要强调一点，实际代码中，start的值是5，表示跳过音频前5帧，这5帧认为是不稳定的。
```

提取的特征已经和视频帧是一一对应了，放入到：
```python
self.feat_queue.put(whisper_chunks)
```

这里总结下，feat_queue是未来送入musetalk要用的音频特征。output_queue是未来要送入webrtc的音频流的数据。output_queue还是按照20ms一帧的，feat_queue已经是40ms一帧了。

这里想说明下，大量的异步线程都是通过queue进行交互的，所以搞清楚每个queue里面有什么很关键。

## 5、musetalk预测
musetalk模型需要的音频特征已经放到feat_queue里面了。
class  MuseReal里面会启动一个inference线程，用来处理模型推理。

```python
class MuseReal:
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        self.render_event.set() #start infer process render
        Thread(target=inference, args=(
            self.render_event, 
            self.batch_size,
            self.input_latent_list_cycle,
            self.asr.feat_queue,
            self.asr.output_queue,
            self.res_frame_queue,
            self.vae, 
            self.unet, 
            self.pe,
            self.timesteps))
        .start()
```
下面就是inference方法:
```python
@torch.no_grad()
def inference(render_event,batch_size,input_latent_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,
              vae, unet, pe,timesteps): #vae, unet, pe,timesteps
    
    length = len(input_latent_list_cycle)
    index = 0
    count=0
    counttime=0
    logger.info('start inference')
    while render_event.is_set():
        starttime=time.perf_counter()
        try:
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence=True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type==0:
                is_all_silence=False
        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # print('infer=======')
            t=time.perf_counter()
            whisper_batch = np.stack(whisper_chunks)
            latent_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # for i, (whisper_batch,latent_batch) in enumerate(gen):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                            dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, 
                                        timesteps, 
                                        encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            counttime += (time.perf_counter() - t)
            count += batch_size
            if count>=100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count=0
                counttime=0
            for i,res_frame in enumerate(recon):
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
    logger.info('musereal inference processor stop')
```
可以看到，首先就是从feature_queue取出一个batch_size 的音频特征。
然后从output_queue取出 batch_size \* 2个音频块。如果对前面各种queue的记录还比较清楚，这里应该能明白，为什么一个是batch_size个，一个是2乘以batch_size个。

然后，根据output_queue中的音频帧的type是不是0判断是不是静音帧，如果是静音帧，就不用进行后面的推理，直接在res_frame_queue放入一个空帧。

这里，我们暂停下，看看系统对静音帧的整体处理流程。

看上面的静音代码，我们知道是根据type是否为0判断是否是静音帧的。type是何时被首次加入的呢？

```python
class BaseASR:
    def get_audio_frame(self):        
        try:
            frame,eventpoint = self.queue.get(block=True,timeout=0.01)
            type = 0
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            if self.parent and self.parent.curr_state>1: #播放自定义音频
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                type = self.parent.curr_state
            else:
                frame = np.zeros(self.chunk, dtype=np.float32)
                type = 1
            eventpoint = None

        return frame,type,eventpoint 
```
可以看到，这里从queue获取音频帧，获取到，则type就是0，如果queue是空的，就会判断能否从self.parent获取音频帧和状态。
self.parent就是MuseReal
```python
class MuseReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar,config):
        super().__init__(opt,config)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        # self.W = opt.W
        # self.H = opt.H

        self.fps = opt.fps # 20 ms per frame

        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size*2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model
        self.frame_list_cycle,self.mask_list_cycle,self.coord_list_cycle,self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar
        #self.__loadavatar()

        self.asr = MuseASR(opt,self,self.audio_processor)
```
最终，`get_audio_stream`就是调用的BaseReal中的对应方法：
```python
class BaseReal:
    def __init__(self, opt,config):
        
        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_img_index = {} #源代码是self.custom_index，为方便理解改为self.custom_img_index
        self.custom_opt = {}
        self.__loadcustom()

    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
```
这里其实就是动作编排的核心代码。简单讲，就是根据curr_state，加载播放对应的音频和视频帧。直接拿来播放。img_cycle和audio_cycle是对应的存储，img_index和audio_index是当前播放到的位置。这几个字典字段的key就是所有可用的curr_state值。


回到我们的inference上，后面就是整理特征， 预测，写入res_frame_queue。这里又出现了一个queue。
其里面的值是：
```python
for i,res_frame in enumerate(recon):
    res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
    index = index + 1
```
元素有三部分组成，第一部分是预测的口型图，第二部分是对应的图片帧的index，第三部分是对应的音频数据。因为是audio_frames是从output_queue中取出的，所以这里要2个。如果觉得有点疑惑，可以返回上面看看各种queue的数据管道。

这里有两点性能优化的地方，需要提一下：
1、res_frame_queue的大小是batch_size * 2 ，限制大小，防止处理太快却播放慢，导致queue容量爆炸。
2、原始的视频帧存储的是index，不是图片数据，音频帧是数据。

## 6、最后拼接视频，将音频和视频送入对应的track队列

MuseReal里面render函数新起一个线程专门处理这个过程。
```python
class MuseReal:
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()
```
上面的`process_frames就是处理拼接返回的。
loop， audio_track, video_track是与webrtc相关的概念， loop是webrtc的主线程循环。audio_track,video_track分别用来将音频和视频放到webrtc队列。
为了能在线程中向webrtc的主线程添加数据，所以一并将loop也当成参数送入线程了。关于webrtc，待会儿专门集中讲解下，这里有点困惑也不要紧，不用深究。

render函数看起来是一个重要入口，我们再来详细看看：
```python
def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        self.render_event.set() #start infer process render
        Thread(target=inference, args=(self.render_event,self.batch_size,self.input_latent_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.vae, self.unet, self.pe,self.timesteps)).start() #mp.Process
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()
```
它自身的while循环用来处理asr run_step方法。
同时，在之前，tts.render启动tts线程。
然后又分别启动了推荐图片回贴，传输给webrtc的线程还有模型推理线程。

那我就比较好奇了，这个render函数是在哪里调用的呢。

我用大模型快速理了一个调用图：

1. 客户端发起 WebRTC 连接
   ↓
2. app.py:offer() 创建 HumanPlayer(nerfreals[sessionid])
   ↓
3. 客户端请求音视频轨道
   ↓
4. PlayerStreamTrack.recv() 被调用
   ↓
5. HumanPlayer._start() 被触发
   ↓
6. 创建工作线程 player_worker_thread
   ↓
7. player_worker_thread 调用 container.render()
   ↓
8. MuseReal.render() 开始执行

其中第4步，是由aiortc框架负责调用的。

又有点跑题，让我们再次聚焦process_frames方法：
```python
class MuseReal:
    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        print(f"[DEBUG] BaseReal process_frames starting - sessionid: {self.opt.sessionid}")
        print(f"[DEBUG] Transport mode: {self.opt.transport}")
        
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
        
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None
            print(f"[DEBUG] Virtual camera mode enabled")

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
            print(f"[DEBUG] Audio thread started for virtual camera")
        
        frame_count = 0
        print(f"[DEBUG] Starting frame processing loop")
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
                frame_count += 1
                
            except queue.Empty:
                print(f"[DEBUG] Frame queue empty, waiting for frames...")
                continue
            
            if enable_transition:
                # 检测状态变化
                current_speaking = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking

            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    print(f"[DEBUG] Initializing virtual camera - width: {width}, height: {height}")
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            else: #webrtc
                image = combine_frame
                image[0,:] &= 0xFE
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            self.record_video_data(combine_frame)

            audio_frame_count = 0
            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)
                audio_frame_count += 1

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes()) #TODO
                else: #webrtc
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                self.record_audio_data(frame)
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()
        if self.opt.transport=='virtualcam':
            audio_thread.join()
            vircam.close()
        logger.info('basereal process_frames thread stop') 
```
这个方法看起来比较复杂，实际逻辑是很简单的，就是从res_frame_queue中取出预测的口型结果、对应原始视频的帧索引，对应的音频数据，将预测结果回贴到原始图片，然后给webrtc返回对应的视频和音频。

需要再次注意一下， res_frame_queue取出的视频预测帧是需要回贴到原始图像上的。得到音频audio_frames长度是2，因为是两个音频帧对应一个视频帧。
每个元素是一个三元组(audio_data, type, eventpoint)
audio_data就是320个音频采样数据，就是一个长度为320的浮点数组。type用来标记类型，0代表说话，1代表静音，大于1的值用来实现动作编排或者其他作用。

这里面添加了一个说话向静音过度和静音向说话过度的功能。
我让claude总结了下：
```python
核心机制

状态检测：通过 audio_frames[0][1]!=0 and audio_frames[1][1]!=0 判断是否为静音状态
过渡触发：状态变化时记录过渡开始时间 _transition_start
帧缓存：保存上一帧的静音帧和说话帧用于混合

过渡实现
说话 → 静音过渡
python# 在静音状态下，如果刚从说话状态切换过来
if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
    alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
    combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
静音 → 说话过渡
python# 在说话状态下，如果刚从静音状态切换过来
if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
    alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
    combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)

关键要点

过渡时长：默认 0.1 秒
混合算法：使用 cv2.addWeighted 按时间比例混合两帧
透明度变化：alpha 值从 0 到 1 线性变化，实现淡入淡出效果
帧缓存更新：过渡完成后更新对应的帧缓存

这样实现了视觉上的平滑过渡，避免了状态切换时的突兀感。
```
这行代码： `current_frame = self.paste_back_frame(res_frame,idx)`
实现了视频回贴。

下面关注视频帧和音频帧的特殊处理：
视频帧有一个：`image[0,:] &= 0xFE` 这样的操作。这看起来是作者留的一个隐形水印，可以检测到使用该系统生成的视频。
音频帧的操作：
```python
 frame = (frame * 32767).astype(np.int16)
```
这不操作是因为frame是tts预测的结果，其取值范围是-1到正1，webrtc对音频的要求是int16,32767恰好是int16可以表示的范围，所以进行一下转换。
以便满足webrtc的要求。
```python
new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
new_frame.planes[0].update(frame.tobytes())  #planes代表声道  
new_frame.sample_rate=16000
```
这里planes代表声道， mono代表单声道， s16代表16有符号数。

## 7、推送到客户浏览器

上面的代码中，处理好的音频和视频数据都通过audio_track和video_track送入各自的队列中。这两个队列在`PlayerStreamTrack`，按照音频20ms一次，视频40ms一次的频率不断取出，送给客户端。
```python
class PlayerStreamTrack(MediaStreamTrack):
    async def recv(self) -> Union[Frame, Packet]:
        
        self._player._start(self)
        
        # 获取帧数据
        # print(f"[DEBUG] Waiting for frame from queue, queue size: {self._queue.qsize()}")
        frame,eventpoint = await self._queue.get()
        
        # 生成时间戳
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        
        # print(f"[DEBUG] Frame received - kind: {self.kind}, pts: {pts}, time_base: {time_base}, eventpoint: {eventpoint}")
        
        if eventpoint:
            print(f"[DEBUG] Notifying player of eventpoint: {eventpoint}")
            self._player.notify(eventpoint)
            
        if frame is None:
            print(f"[DEBUG] Received null frame, stopping track")
            self.stop()
            raise Exception
            
        if self.kind == 'video':
            current_time = time.perf_counter()
            self.totaltime += (current_time - self.lasttime)
            self.framecount += 1
            self.lasttime = current_time
            
            if self.framecount==100:
                fps = self.framecount/self.totaltime
                print(f"[DEBUG] Video FPS report - frames: {self.framecount}, avg_fps: {fps:.4f}")
                mylogger.info(f"------actual avg final fps:{fps:.4f}")
                self.framecount = 0
                self.totaltime=0
        else:
            # print(f"[DEBUG] Audio frame processed - samples: {frame.samples if hasattr(frame, 'samples') else 'N/A'}")
            pass
            
        return frame
```
这个方法是被aiortc自动调用的。

`pts, time_base = await self.next_timestamp()`这行代码是确保视频按照40ms一帧，音频按照20ms一帧同步推送的关键所在。

```python
async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                old_timestamp = self._timestamp
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                current_time = time.time()
                wait = self._start + self.current_frame_count * VIDEO_PTIME - current_time
                
                if wait>0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                old_timestamp = self._timestamp
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                current_time = time.time()
                wait = self._start + self.current_frame_count * AUDIO_PTIME - current_time
                
                if wait>0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
            return self._timestamp, AUDIO_TIME_BASE
```
通过`self._start + self.current_frame_count * VIDEO_PTIME` 精确计算该帧的推送时间，然后和`current_time`比较，如果时间还没到，即wait大于0，则进行等待。
这样就通过严格是时间对齐，确保音频视频同步。

recv方法是如何被aiortc调用的呢。
在app.py文件中有一个run方法：
```python
async def run(push_url,sessionid):
    print(f"[DEBUG] Starting RTC push for sessionid: {sessionid}, push_url: {push_url}")
    
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    print(f"[DEBUG] Built nerfreal for push sessionid: {sessionid}")

    pc = RTCPeerConnection()
    pcs.add(pc)
    print(f"[DEBUG] Created RTCPeerConnection for push, total connections: {len(pcs)}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[DEBUG] Push connection state changed to: {pc.connectionState} for sessionid: {sessionid}")
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            print(f"[DEBUG] Push connection failed, cleaning up sessionid: {sessionid}")
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    print(f"[DEBUG] Created HumanPlayer for push sessionid: {sessionid}")
    
    audio_sender = pc.addTrack(player.audio)
    print(f"[DEBUG] Added audio track for push: {audio_sender}")
    
    video_sender = pc.addTrack(player.video)
```
这里pc.addTrack就将音频和视频track加入了aiortc，它将不断调用recv获取数据，根据时间戳发送。 recv这个方法签名来自aiortc中的MediaStreamTrack。PlayerStreamTrack继承了这个类并实现了该方法。


## 8、webrtc创建连接的过程

这个项目中，大家使用反馈最多就是webrtc连接不上。这里梳理下连接创建的过程。


1. 客户端发起连接请求
客户端通过POST请求到/offer端点，发送包含SDP offer的JSON数据：

```json
{
  "sdp": "客户端生成的SDP offer",
  "type": "offer"
}
```
给一个具体实例：
```
 {'sdp': 'v=0\r\no=- 1823584063176897508 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0 1\r\na=extmap-allow-mixed\r\na=msid-semantic: WMS\r\nm=video 39224 UDP/TLS/RTP/SAVPF 96 97 98 99 100 101 35 36 37 38 103 104 107 108 109 114 115 116 117 118 39 40 41 42 43 44 45 46 47 48 119 120 121 122 49 50 51 52 123 124 125 53\r\nc=IN IP4 120.244.192.143\r\na=rtcp:9 IN IP4 0.0.0.0\r\na=candidate:3979796857 1 udp 2113937151 6693c814-ec13-4db3-b14d-8c1aafbccbc1.local 60370 typ host generation 0 network-cost 999\r\na=candidate:3182964466 1 udp 2113939711 81b154c7-3930-4ee2-b39f-b867f033e148.local 51489 typ host generation 0 network-cost 999\r\na=candidate:153078608 1 udp 1677729535 120.244.192.143 39224 typ srflx raddr 0.0.0.0 rport 0 generation 0 network-cost 999\r\na=candidate:2064463779 1 udp 1677732095 2409:8a00:dc3:8dd0:8893:5dfe:3908:c91c 51489 typ srflx raddr :: rport 0 generation 0 network-cost 999\r\na=ice-ufrag:mLgr\r\na=ice-pwd:CQ7SY9kDBVBiofWGhjL2nEeT\r\na=ice-options:trickle\r\na=fingerprint:sha-256 79:FA:0B:A9:58:1C:AF:34:C7:05:D9:A8:1F:B9:8E:06:D6:B4:4B:6E:31:A5:39:F0:3C:EC:53:B8:08:0F:43:1A\r\na=setup:actpass\r\na=mid:0\r\na=extmap:1 urn:ietf:params:rtp-hdrext:toffset\r\na=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time\r\na=extmap:3 urn:3gpp:video-orientation\r\na=extmap:4 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01\r\na=extmap:5 http://www.webrtc.org/experiments/rtp-hdrext/playout-delay\r\na=extmap:6 http://www.webrtc.org/experiments/rtp-hdrext/video-content-type\r\na=extmap:7 http://www.webrtc.org/experiments/rtp-hdrext/video-timing\r\na=extmap:8 http://www.webrtc.org/experiments/rtp-hdrext/color-space\r\na=extmap:9 urn:ietf:params:rtp-hdrext:sdes:mid\r\na=extmap:10 urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id\r\na=extmap:11 urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id\r\na=recvonly\r\na=rtcp-mux\r\na=rtcp-rsize\r\na=rtpmap:96 VP8/90000\r\na=rtcp-fb:96 goog-remb\r\na=rtcp-fb:96 transport-cc\r\na=rtcp-fb:96 ccm fir\r\na=rtcp-fb:96 nack\r\na=rtcp-fb:96 nack pli\r\na=rtpmap:97 rtx/90000\r\na=fmtp:97 apt=96\r\na=rtpmap:98 VP9/90000\r\na=rtcp-fb:98 goog-remb\r\na=rtcp-fb:98 transport-cc\r\na=rtcp-fb:98 ccm fir\r\na=rtcp-fb:98 nack\r\na=rtcp-fb:98 nack pli\r\na=fmtp:98 profile-id=0\r\na=rtpmap:99 rtx/90000\r\na=fmtp:99 apt=98\r\na=rtpmap:100 VP9/90000\r\na=rtcp-fb:100 goog-remb\r\na=rtcp-fb:100 transport-cc\r\na=rtcp-fb:100 ccm fir\r\na=rtcp-fb:100 nack\r\na=rtcp-fb:100 nack pli\r\na=fmtp:100 profile-id=2\r\na=rtpmap:101 rtx/90000\r\na=fmtp:101 apt=100\r\na=rtpmap:35 VP9/90000\r\na=rtcp-fb:35 goog-remb\r\na=rtcp-fb:35 transport-cc\r\na=rtcp-fb:35 ccm fir\r\na=rtcp-fb:35 nack\r\na=rtcp-fb:35 nack pli\r\na=fmtp:35 profile-id=1\r\na=rtpmap:36 rtx/90000\r\na=fmtp:36 apt=35\r\na=rtpmap:37 VP9/90000\r\na=rtcp-fb:37 goog-remb\r\na=rtcp-fb:37 transport-cc\r\na=rtcp-fb:37 ccm fir\r\na=rtcp-fb:37 nack\r\na=rtcp-fb:37 nack pli\r\na=fmtp:37 profile-id=3\r\na=rtpmap:38 rtx/90000\r\na=fmtp:38 apt=37\r\na=rtpmap:103 H264/90000\r\na=rtcp-fb:103 goog-remb\r\na=rtcp-fb:103 transport-cc\r\na=rtcp-fb:103 ccm fir\r\na=rtcp-fb:103 nack\r\na=rtcp-fb:103 nack pli\r\na=fmtp:103 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f\r\na=rtpmap:104 rtx/90000\r\na=fmtp:104 apt=103\r\na=rtpmap:107 H264/90000\r\na=rtcp-fb:107 goog-remb\r\na=rtcp-fb:107 transport-cc\r\na=rtcp-fb:107 ccm fir\r\na=rtcp-fb:107 nack\r\na=rtcp-fb:107 nack pli\r\na=fmtp:107 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42001f\r\na=rtpmap:108 rtx/90000\r\na=fmtp:108 apt=107\r\na=rtpmap:109 H264/90000\r\na=rtcp-fb:109 goog-remb\r\na=rtcp-fb:109 transport-cc\r\na=rtcp-fb:109 ccm fir\r\na=rtcp-fb:109 nack\r\na=rtcp-fb:109 nack pli\r\na=fmtp:109 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f\r\na=rtpmap:114 rtx/90000\r\na=fmtp:114 apt=109\r\na=rtpmap:115 H264/90000\r\na=rtcp-fb:115 goog-remb\r\na=rtcp-fb:115 transport-cc\r\na=rtcp-fb:115 ccm fir\r\na=rtcp-fb:115 nack\r\na=rtcp-fb:115 nack pli\r\na=fmtp:115 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f\r\na=rtpmap:116 rtx/90000\r\na=fmtp:116 apt=115\r\na=rtpmap:117 H264/90000\r\na=rtcp-fb:117 goog-remb\r\na=rtcp-fb:117 transport-cc\r\na=rtcp-fb:117 ccm fir\r\na=rtcp-fb:117 nack\r\na=rtcp-fb:117 nack pli\r\na=fmtp:117 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=4d001f\r\na=rtpmap:118 rtx/90000\r\na=fmtp:118 apt=117\r\na=rtpmap:39 H264/90000\r\na=rtcp-fb:39 goog-remb\r\na=rtcp-fb:39 transport-cc\r\na=rtcp-fb:39 ccm fir\r\na=rtcp-fb:39 nack\r\na=rtcp-fb:39 nack pli\r\na=fmtp:39 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=4d001f\r\na=rtpmap:40 rtx/90000\r\na=fmtp:40 apt=39\r\na=rtpmap:41 H264/90000\r\na=rtcp-fb:41 goog-remb\r\na=rtcp-fb:41 transport-cc\r\na=rtcp-fb:41 ccm fir\r\na=rtcp-fb:41 nack\r\na=rtcp-fb:41 nack pli\r\na=fmtp:41 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=f4001f\r\na=rtpmap:42 rtx/90000\r\na=fmtp:42 apt=41\r\na=rtpmap:43 H264/90000\r\na=rtcp-fb:43 goog-remb\r\na=rtcp-fb:43 transport-cc\r\na=rtcp-fb:43 ccm fir\r\na=rtcp-fb:43 nack\r\na=rtcp-fb:43 nack pli\r\na=fmtp:43 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=f4001f\r\na=rtpmap:44 rtx/90000\r\na=fmtp:44 apt=43\r\na=rtpmap:45 AV1/90000\r\na=rtcp-fb:45 goog-remb\r\na=rtcp-fb:45 transport-cc\r\na=rtcp-fb:45 ccm fir\r\na=rtcp-fb:45 nack\r\na=rtcp-fb:45 nack pli\r\na=fmtp:45 level-idx=5;profile=0;tier=0\r\na=rtpmap:46 rtx/90000\r\na=fmtp:46 apt=45\r\na=rtpmap:47 AV1/90000\r\na=rtcp-fb:47 goog-remb\r\na=rtcp-fb:47 transport-cc\r\na=rtcp-fb:47 ccm fir\r\na=rtcp-fb:47 nack\r\na=rtcp-fb:47 nack pli\r\na=fmtp:47 level-idx=5;profile=1;tier=0\r\na=rtpmap:48 rtx/90000\r\na=fmtp:48 apt=47\r\na=rtpmap:119 H264/90000\r\na=rtcp-fb:119 goog-remb\r\na=rtcp-fb:119 transport-cc\r\na=rtcp-fb:119 ccm fir\r\na=rtcp-fb:119 nack\r\na=rtcp-fb:119 nack pli\r\na=fmtp:119 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=64001f\r\na=rtpmap:120 rtx/90000\r\na=fmtp:120 apt=119\r\na=rtpmap:121 H264/90000\r\na=rtcp-fb:121 goog-remb\r\na=rtcp-fb:121 transport-cc\r\na=rtcp-fb:121 ccm fir\r\na=rtcp-fb:121 nack\r\na=rtcp-fb:121 nack pli\r\na=fmtp:121 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=64001f\r\na=rtpmap:122 rtx/90000\r\na=fmtp:122 apt=121\r\na=rtpmap:49 H265/90000\r\na=rtcp-fb:49 goog-remb\r\na=rtcp-fb:49 transport-cc\r\na=rtcp-fb:49 ccm fir\r\na=rtcp-fb:49 nack\r\na=rtcp-fb:49 nack pli\r\na=fmtp:49 level-id=180;profile-id=1;tier-flag=0;tx-mode=SRST\r\na=rtpmap:50 rtx/90000\r\na=fmtp:50 apt=49\r\na=rtpmap:51 H265/90000\r\na=rtcp-fb:51 goog-remb\r\na=rtcp-fb:51 transport-cc\r\na=rtcp-fb:51 ccm fir\r\na=rtcp-fb:51 nack\r\na=rtcp-fb:51 nack pli\r\na=fmtp:51 level-id=180;profile-id=2;tier-flag=0;tx-mode=SRST\r\na=rtpmap:52 rtx/90000\r\na=fmtp:52 apt=51\r\na=rtpmap:123 red/90000\r\na=rtpmap:124 rtx/90000\r\na=fmtp:124 apt=123\r\na=rtpmap:125 ulpfec/90000\r\na=rtpmap:53 flexfec-03/90000\r\na=rtcp-fb:53 goog-remb\r\na=rtcp-fb:53 transport-cc\r\na=fmtp:53 repair-window=10000000\r\nm=audio 39225 UDP/TLS/RTP/SAVPF 111 63 9 0 8 13 110 126\r\nc=IN IP4 120.244.192.143\r\na=rtcp:9 IN IP4 0.0.0.0\r\na=candidate:3979796857 1 udp 2113937151 6693c814-ec13-4db3-b14d-8c1aafbccbc1.local 64300 typ host generation 0 network-cost 999\r\na=candidate:3182964466 1 udp 2113939711 81b154c7-3930-4ee2-b39f-b867f033e148.local 63169 typ host generation 0 network-cost 999\r\na=candidate:2064463779 1 udp 1677732095 2409:8a00:dc3:8dd0:8893:5dfe:3908:c91c 63169 typ srflx raddr :: rport 0 generation 0 network-cost 999\r\na=candidate:153078608 1 udp 1677729535 120.244.192.143 39225 typ srflx raddr 0.0.0.0 rport 0 generation 0 network-cost 999\r\na=ice-ufrag:mLgr\r\na=ice-pwd:CQ7SY9kDBVBiofWGhjL2nEeT\r\na=ice-options:trickle\r\na=fingerprint:sha-256 79:FA:0B:A9:58:1C:AF:34:C7:05:D9:A8:1F:B9:8E:06:D6:B4:4B:6E:31:A5:39:F0:3C:EC:53:B8:08:0F:43:1A\r\na=setup:actpass\r\na=mid:1\r\na=extmap:14 urn:ietf:params:rtp-hdrext:ssrc-audio-level\r\na=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time\r\na=extmap:4 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01\r\na=extmap:9 urn:ietf:params:rtp-hdrext:sdes:mid\r\na=recvonly\r\na=rtcp-mux\r\na=rtcp-rsize\r\na=rtpmap:111 opus/48000/2\r\na=rtcp-fb:111 transport-cc\r\na=fmtp:111 minptime=10;useinbandfec=1\r\na=rtpmap:63 red/48000/2\r\na=fmtp:63 111/111\r\na=rtpmap:9 G722/8000\r\na=rtpmap:0 PCMU/8000\r\na=rtpmap:8 PCMA/8000\r\na=rtpmap:13 CN/8000\r\na=rtpmap:110 telephone-event/48000\r\na=rtpmap:126 telephone-event/8000\r\n', 'type': 'offer'}
```
看着很乱，就重点关注`candidate:`字段即可，这些就是客户端，通过stun服务器和本地网口，生成的各种可以用来创建连接的地址。

2. 服务器端处理流程
2.1 接收和解析Offer
```python
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
```
解析客户端发送的JSON参数

创建RTCSessionDescription对象，包含客户端的SDP offer

2.2 会话管理

```python
sessionid = randN(6)  # 生成6位随机会话ID
nerfreals[sessionid] = None
nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
nerfreals[sessionid] = nerfreal
```
生成唯一的6位随机会话ID

根据配置的模型类型（musetalk/wav2lip/ultralight）创建对应的AI模型实例

将模型实例存储在全局字典中。

这里需要注意，不同的session共享模型实例，而不是每个session单独加载一套模型，那样很快就炸了。

2.3 创建PeerConnection
```python
ice_server = RTCIceServer(urls='stun:stun.miwifi.com:3478')
pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
pcs.add(pc)
```
这里的ice_server是用来帮助服务器发现自己的公网可访问地址的。

配置STUN服务器用于NAT穿透

创建RTCPeerConnection对象

将连接添加到全局连接集合中



2.4 信令交换
```python
await pc.setRemoteDescription(offer)  # 设置远程描述（客户端offer）
answer = await pc.createAnswer()      # 创建answer
await pc.setLocalDescription(answer)  # 设置本地描述

```
这里的answer是服务器端发现的自己的连接candidates
```
[DEBUG] Server SDP: v=0
o=- 3961364075 3961364075 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic:WMS *
m=video 33618 UDP/TLS/RTP/SAVPF 103 104 109 114 96 97
c=IN IP4 192.168.1.17
a=sendonly
a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time
a=extmap:9 urn:ietf:params:rtp-hdrext:sdes:mid
a=mid:0
a=msid:ae726652-6e17-40f6-90e8-aa02ef550ee8 4946f8ec-1a25-4acd-bd39-9867fcb2c5ad
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc-group:FID 111276156 1803568782
a=ssrc:111276156 cname:338d1c26-3bdf-4980-8593-977179b0d0bf
a=ssrc:1803568782 cname:338d1c26-3bdf-4980-8593-977179b0d0bf
a=rtpmap:103 H264/90000
a=rtcp-fb:103 nack
a=rtcp-fb:103 nack pli
a=rtcp-fb:103 goog-remb
a=fmtp:103 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f
a=rtpmap:104 rtx/90000
a=fmtp:104 apt=103
a=rtpmap:109 H264/90000
a=rtcp-fb:109 nack
a=rtcp-fb:109 nack pli
a=rtcp-fb:109 goog-remb
a=fmtp:109 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
a=rtpmap:114 rtx/90000
a=fmtp:114 apt=109
a=rtpmap:96 VP8/90000
a=rtcp-fb:96 nack
a=rtcp-fb:96 nack pli
a=rtcp-fb:96 goog-remb
a=rtpmap:97 rtx/90000
a=fmtp:97 apt=96
a=candidate:97335fc4d7cfcbb401af0e4689511738 1 udp 2130706431 192.168.1.17 33618 typ host
a=candidate:e6a2fda0b61d6fb7ba0b961ae9bc2e77 1 udp 2130706431 2408:821b:3522:2aa0:e39f:eb71:8fe0:3ca3 34320 typ host
a=candidate:fdb8a42c8528152d19b65cc83c773ea4 1 udp 2130706431 2408:821b:3522:2aa0:a0ad:f26c:3559:cfc3 56200 typ host
a=candidate:a905fcbaee4e1f834d1a3a62113be28d 1 udp 2130706431 2408:821b:3522:2aa0:42d9:ee7a:ae9a:7ecc 33431 typ host
a=candidate:3d1e92e022cccf7e926f6a452efb1cad 1 udp 2130706431 2408:821b:3522:2aa0:e2c5:de4b:7ac6:e96d 40183 typ host
a=candidate:a2beb7ab6877df1064a63fbecec70611 1 udp 2130706431 2408:821b:3522:2aa0:a02c:5f56:479e:ae2d 45462 typ host
a=candidate:761eec014d151d9480f97e2a9eeaa024 1 udp 2130706431 2408:821b:3522:2aa0:6d68:173a:8884:5205 48831 typ host
a=candidate:864202f60b3b402042862864ae1d1a93 1 udp 2130706431 2408:821b:3522:2aa0:3259:e477:1f9c:cc93 50786 typ host
a=candidate:d76dab0e3cc970e303c5a399230f8f78 1 udp 2130706431 2408:821b:3522:2aa0:11a1:b4dd:b275:a8e7 36795 typ host
a=candidate:93731beb880374226c8db07cd1cc7b1e 1 udp 2130706431 2408:821b:3522:2aa0:10b4:a446:7215:77f4 47220 typ host
a=candidate:580b5f32035da7f14a30ea7a8d826c67 1 udp 2130706431 172.17.0.1 40681 typ host
a=candidate:a8865e46e09c472a25b40fa2700b6911 1 udp 2130706431 192.168.16.1 48973 typ host
a=candidate:0ad8db753210ce9554cf251a9e51e792 1 udp 1694498815 110.248.21.154 47423 typ srflx raddr 172.17.0.1 rport 40681
a=candidate:109b8b9adbaa17f659d6c16091cc9c94 1 udp 1694498815 110.248.21.154 47872 typ srflx raddr 192.168.1.17 rport 33618
a=candidate:1f6c29ae00129bb8a7e0c52333523782 1 udp 1694498815 110.248.21.154 47422 typ srflx raddr 192.168.16.1 rport 48973
a=end-of-candidates
a=ice-ufrag:1J7r
a=ice-pwd:KrKOoVNexiQYONQgKSPNCr
a=fingerprint:sha-256 8A:A6:BB:8C:A8:5C:6D:5F:35:64:8D:51:17:4B:67:68:1F:76:38:72:83:A3:D0:29:3D:76:EE:81:03:8B:B5:CA
a=fingerprint:sha-384 27:74:2D:F0:94:42:CE:13:65:9D:63:48:98:CE:28:8B:F7:F2:C5:7E:07:17:39:AA:27:7E:88:2B:A6:0F:5B:0E:A1:DD:73:6F:F0:8F:DB:3C:01:FD:E8:77:1E:0E:9A:15
a=fingerprint:sha-512 E2:78:87:44:85:4B:45:67:6D:71:BE:4E:D7:AF:D1:59:D6:3E:9C:2B:E6:18:33:E3:5B:10:81:0A:BA:F8:BB:1D:75:B4:2D:55:E2:2D:88:42:E6:FD:BF:53:7B:02:73:E9:D5:74:B5:92:3E:49:3D:1E:58:35:9F:80:38:87:A2:56
a=setup:active
m=audio 33618 UDP/TLS/RTP/SAVPF 111 9 0 8
c=IN IP4 192.168.1.17
a=sendonly
a=extmap:14 urn:ietf:params:rtp-hdrext:ssrc-audio-level
a=extmap:9 urn:ietf:params:rtp-hdrext:sdes:mid
a=mid:1
a=msid:ae726652-6e17-40f6-90e8-aa02ef550ee8 1192591b-d551-4945-9d2e-1b9409069233
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc:655098292 cname:338d1c26-3bdf-4980-8593-977179b0d0bf
a=rtpmap:111 opus/48000/2
a=rtpmap:9 G722/8000
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=candidate:97335fc4d7cfcbb401af0e4689511738 1 udp 2130706431 192.168.1.17 33618 typ host
a=candidate:e6a2fda0b61d6fb7ba0b961ae9bc2e77 1 udp 2130706431 2408:821b:3522:2aa0:e39f:eb71:8fe0:3ca3 34320 typ host
a=candidate:fdb8a42c8528152d19b65cc83c773ea4 1 udp 2130706431 2408:821b:3522:2aa0:a0ad:f26c:3559:cfc3 56200 typ host
a=candidate:a905fcbaee4e1f834d1a3a62113be28d 1 udp 2130706431 2408:821b:3522:2aa0:42d9:ee7a:ae9a:7ecc 33431 typ host
a=candidate:3d1e92e022cccf7e926f6a452efb1cad 1 udp 2130706431 2408:821b:3522:2aa0:e2c5:de4b:7ac6:e96d 40183 typ host
a=candidate:a2beb7ab6877df1064a63fbecec70611 1 udp 2130706431 2408:821b:3522:2aa0:a02c:5f56:479e:ae2d 45462 typ host
a=candidate:761eec014d151d9480f97e2a9eeaa024 1 udp 2130706431 2408:821b:3522:2aa0:6d68:173a:8884:5205 48831 typ host
a=candidate:864202f60b3b402042862864ae1d1a93 1 udp 2130706431 2408:821b:3522:2aa0:3259:e477:1f9c:cc93 50786 typ host
a=candidate:d76dab0e3cc970e303c5a399230f8f78 1 udp 2130706431 2408:821b:3522:2aa0:11a1:b4dd:b275:a8e7 36795 typ host
a=candidate:93731beb880374226c8db07cd1cc7b1e 1 udp 2130706431 2408:821b:3522:2aa0:10b4:a446:7215:77f4 47220 typ host
a=candidate:580b5f32035da7f14a30ea7a8d826c67 1 udp 2130706431 172.17.0.1 40681 typ host
a=candidate:a8865e46e09c472a25b40fa2700b6911 1 udp 2130706431 192.168.16.1 48973 typ host
a=candidate:0ad8db753210ce9554cf251a9e51e792 1 udp 1694498815 110.248.21.154 47423 typ srflx raddr 172.17.0.1 rport 40681
a=candidate:109b8b9adbaa17f659d6c16091cc9c94 1 udp 1694498815 110.248.21.154 47872 typ srflx raddr 192.168.1.17 rport 33618
a=candidate:1f6c29ae00129bb8a7e0c52333523782 1 udp 1694498815 110.248.21.154 47422 typ srflx raddr 192.168.16.1 rport 48973
a=end-of-candidates
a=ice-ufrag:1J7r
a=ice-pwd:KrKOoVNexiQYONQgKSPNCr
a=fingerprint:sha-256 8A:A6:BB:8C:A8:5C:6D:5F:35:64:8D:51:17:4B:67:68:1F:76:38:72:83:A3:D0:29:3D:76:EE:81:03:8B:B5:CA
a=fingerprint:sha-384 27:74:2D:F0:94:42:CE:13:65:9D:63:48:98:CE:28:8B:F7:F2:C5:7E:07:17:39:AA:27:7E:88:2B:A6:0F:5B:0E:A1:DD:73:6F:F0:8F:DB:3C:01:FD:E8:77:1E:0E:9A:15
a=fingerprint:sha-512 E2:78:87:44:85:4B:45:67:6D:71:BE:4E:D7:AF:D1:59:D6:3E:9C:2B:E6:18:33:E3:5B:10:81:0A:BA:F8:BB:1D:75:B4:2D:55:E2:2D:88:42:E6:FD:BF:53:7B:02:73:E9:D5:74:B5:92:3E:49:3D:1E:58:35:9F:80:38:87:A2:56
a=setup:active
```

返回给客户端:
```python
response_data = {
    "sdp": pc.localDescription.sdp, 
    "type": pc.localDescription.type, 
    "sessionid": sessionid
}
return web.Response(content_type="application/json", text=json.dumps(response_data))
```
这样，二者就是按照各自的candidates的优先级尝试连接。

```python
@pc.on("connectionstatechange")
async def on_connectionstatechange():
    # 处理连接状态变化
    
@pc.on("icegatheringstatechange") 
async def on_icegatheringstatechange():
    # 处理ICE候选收集状态变化
    
@pc.on("icecandidate")
async def on_icecandidate(event):
    # 处理ICE候选收集
```
这几个装饰器方法，用来监控连接建立过程中的事件。


## 9、总结
以上，就是对livetalking项目运行过程的一个总结，以备忘查。


















