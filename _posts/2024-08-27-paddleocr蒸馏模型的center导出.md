---
layout: post
title:  "paddleocr蒸馏模型的center导出"
date:   2024-08-27 09:20:08 +0800
category: "AI"
published: false
---

最近使用paddleocr框架训练一个识别模型，方式是蒸馏svtrv2。训练完成后，想进一步实验一下添加center loss，对识别形近字的提升效果。
第一步就是要到处center，官方文档中只有简单的ocrv2的center导出。没有蒸馏模型，多head的情况下的导出。把自己的解决方案记录下来。
<!--more-->

# 一、修改配置文件
蒸馏训练使用的配置文件来自rec_svtrv2_ch_distillation.yml。
为了能够导出center，需要修改一下里面的ctc head的配置：
原来的配置：
```yaml
Head:
        name: MultiHead
        head_list:
          - CTCHead:
              Neck:
                name: svtr
                dims: 256
                depth: 2
                hidden_dims: 256
                kernel_size: [1, 3]
                use_guide: True
              Head:
                fc_decay: 0.00001
          - NRTRHead:
              nrtr_dim: 384
              num_decoder_layers: 2
              max_text_length: *max_text_length
```
修改后的配置：
```yaml
Head:
        name: MultiHead
        head_list:
          - CTCHead:
              Neck:
                name: svtr
                dims: 256
                depth: 2
                hidden_dims: 256
                kernel_size: [1, 3]
                use_guide: True
              Head:
                fc_decay: 0.00001
                return_feats: true  #这是修改添加的部分
          - NRTRHead:
              nrtr_dim: 384
              num_decoder_layers: 2
              max_text_length: *max_text_length
```
修改后，要求ctc head 返回 feats和logits。

# 二、修改export_center.py
主要有三处改动。第一是修改datasetname

原先的代码：

```python
config["Eval"]["dataset"]["name"] = config["Train"]["dataset"]["name"]
config["Eval"]["dataset"]["data_dir"] = config["Train"]["dataset"]["data_dir"]
config["Eval"]["dataset"]["label_file_list"] = config["Train"]["dataset"][
    "label_file_list"
]
```
修改后的代码

```python
# config["Eval"]["dataset"]["name"] = config["Train"]["dataset"]["name"]
config["Eval"]["dataset"]["data_dir"] = config["Train"]["dataset"]["data_dir"]
config["Eval"]["dataset"]["label_file_list"] = config["Train"]["dataset"][
    "label_file_list"
]
```
这里注释掉了dataset的name，因为蒸馏训练的配置中，训练时使用的是MultiScaleDataSet，评估时使用的是SimpleDataSet。
因为是只需要导出center，所以这里还是使用SimpleDataSet。

第二是在config中添加out_channels_list，因为不添加一直报错。官方repo的issues上有不少这个报错，但最后都没有明确的解决。笔者就想，为什么训练的时候就没有报错呢？
于是打开train.py文件，原来是在这里添加上的，所以直接将相应代码，复制粘贴到export_center.py中。
如下所示：
```python

# build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    # update SARLoss params
                    if (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif any(
                        "DistillationNRTRLoss" in d
                        for d in config["Loss"]["loss_config_list"]
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            # update SARLoss params
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode":  # for SAR model
            config["Loss"]["ignore_index"] = char_num - 1
```
添加的位置就在build_model调用之前。

第三是修改模型为eval模式。
因为蒸馏训练时有nrtr loss，训练模式下，需要输入label以便生成gtc head。但这个head在导出center时是没有用的。
而且dataset是eval模式，送入模型的数据也没有label，所以要修改成eval模式，不然会报错。
具体来讲，就是在
```model = build_model(config["Architecture"])```
后面添加上：
```model.eval()```


# 三、修改rec_ctc_head.py
为什么必须修改这里的源码呢？前面因为把模型设置为eval模式，规避gtc head 报错。
但是ctc head的实现中，如果是eval模式，则不会返回center需要的features。
代码如下：
```python
def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
            result = predicts

        return result
```
笔者也没想到更好的办法，只能临时在里面设置一下。
设置后的代码：
```python
def forward(self, x, targets=None):
        self.training = True #临时添加的代码
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
            result = predicts

        return result
```
切记：这只是为了导出center临时添加，一旦center 导出完毕，还需要恢复这里的代码。

# 四、总结
至此，笔者成功实现了导出center的程序。paddleocr在快速发展中，很多代码之间缺乏兼容性，使用起来有一定的门槛，对新手并不是很友好。
官方对解决这些问题，更多依赖社区。总的来说，对学习ocr，这个库是不可能绕过去的了。
