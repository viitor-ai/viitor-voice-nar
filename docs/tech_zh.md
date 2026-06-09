# ViiTorVoice-NAR 技术说明

## 模型结构与特性

### 模型结构

ViiTorVoice 采用非自回归的离散 masked language model。模型不按自回归方式逐 token 生成，而是在离散音频 token 空间中对 masked codebook 进行补全。

音频表示使用 DualCodec 25Hz 的 12 层 codebook。12 层离散 codebook 同时承载语义和声学信息，配合模型的 masked LM 结构完成语音克隆、局部编辑和条件控制。

### 局部编辑

局部编辑任务输入原始音频、原始文本和修改后的完整文本。服务会先基于原始文本与编辑后文本做 diff，再结合对齐结果确定需要替换的音频区域，最后在上下文和文本条件约束下补全被 mask 的音频 token。

这种方式不需要整段音频重新生成，更适合修改短词、短句、数字、人名、语气词等局部内容。

### 情感控制与副语言信息插入

模型支持在文本条件中插入情感信息与副语言信息，例如情绪标签、语气、笑声、停顿或其他表达方式。推理时可以通过 `emotion_guidance_scale` 和 `nvv_guidance_scale` 等 CFG 参数增强对应条件的影响。

当文本中没有对应标签时，相关 CFG 参数会被忽略；当存在标签时，较高的 guidance scale 通常会让控制信号更明显，但也需要结合音质和自然度调节。

### No-ref-text 语音克隆

ViiTorVoice 支持 no-ref-text 克隆。调用语音克隆接口时，可以只提供提示音频，不提供提示音对应文本；模型会按 no-ref-text 逻辑处理参考音频，从而降低使用门槛。

HTTP 接口中 `ref_text` 默认为空字符串，`allow_missing_ref_text` 默认为 `true`，可直接用于无提示文本的音色克隆场景。

### First Block 低延迟推理

ViiTorVoice 支持 first block 推理模式：在给定音频总长与 first block 长度的情况下，模型只生成 first block 中的 token，用于加速首帧生成。

该模式面向低延迟合成和流式体验，端到端首帧返回时间可以做到约 60ms。后续音频块可以继续按分块方式生成，从而兼顾首帧响应速度和完整音频质量。

## 技术做法

ViiTorVoice 的模型结构很大程度参考了 OmniVoice。核心不同点主要体现在训练任务、音频 codebook、分词方式和后训练策略上。

### 训练任务扩展

除常规语音生成任务外，训练中额外加入了两个分支。

第一是 first block 模式。训练时给定音频总长与 first block 长度，只要求模型生成 first block 中的 token，让模型显式学习低延迟首块生成能力。

第二是 edit 模式。训练时随机选择一段连续音频 token 做 mask，模型需要根据文本条件、mask 前后的音频上下文以及编辑条件补全该片段。这个分支直接对应推理阶段的局部编辑能力。

### DualCodec Codebook

项目使用 DualCodec 作为音频 codebook 来源。相比单一混合表示，DualCodec 更明确地区分 semantic 和 acoustic 信息，有利于模型同时学习内容一致性、音色保持和声学细节恢复。

当前使用的是 25Hz、12 层 codebook。25Hz 降低了时间维 token 密度，12 层结构提供了足够的离散表达能力，适合在质量和推理效率之间取得平衡。

### 分语种 Tokenizer

文本 tokenizer 按语种区分。中、日、韩等语言使用字符级分词，以减少分词歧义对对齐和局部编辑的影响；英文等语言可以使用更适合该语种的词级或子词级表示。

在局部编辑中，中文、日文、韩文默认使用字符级 align，英文默认使用词级 align。HTTP 接口可以通过 `align_granularity` 覆盖自动选择。

### 一致性蒸馏与 CFG 学习

后训练阶段使用一致性蒸馏，让模型在更少推理步数下保持生成质量。同时，训练中明确学习 CFG 模式，使模型在情感控制、副语言控制和其他条件增强场景下更稳定。

这一策略进一步减少推理步数，提升端到端推理速度，并降低低延迟场景中的首帧等待时间。
