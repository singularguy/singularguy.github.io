# OpenRLHF源码解读&1.理解PPO单机训练
> _**作者: 姜富春**_
> 
> _**原文:**_ [_**https://zhuanlan.zhihu.com/p/13043187674**_](https://zhuanlan.zhihu.com/p/13043187674)

0.OpenRLHF简介
------------

本人对PPO一直停留在“理论”和“实践”层面， 看过PPO的原理，训过PPO的模型，但一直没有从源码角度深入理解PPO实现，相信跟我一样的人不少。原因是始终也没找到一个合适入手的代码，直到看到了OpenRLHF，代码简洁，目录清晰，让我迫不及待想深入理解下细节，学习过程中整理了这篇文档，希望跟大家一起交流学习~

本文从**单机的PPO源码**入手，串起来PPO训练[语言模型](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)的细节。

1.源码概要
------

源码地址：[OpenRLHF](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/README_zh.md)

PPO论文：[https://arxiv.org/pdf/1707.06347](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1707.06347)

### 1.1代码结构

```text-plain
OpenRLHF
|--examples                    //示例启动脚本
|----scripts
|------train_ppo_llama.sh      //训练PPO 
|------train_sft_llama.sh      //SFT
|------train_rm_llama.sh       //训练reward model 
|------......                  //还有很多 包括其他训练方法、分布式训练等
|--openrlhf                    //核心代码块
|----cli                       //训练入口函数
|----datasets                  //数据集处理相关
|----models                    //定义模型、loss相关
|----trainer                   //定义训练方法
|----utils                     //工具类、函数定义
```

### 1.2 单机PPO源码解读

OpenRLHF提供了多种Post-training方法，本文只围绕**PPO相关源码**做解读

PPO训练入口：`OpenRLHF/examples/scripts/train_ppo_llama.sh`

**首先通过一张图概述PPO训练的全过程。**

![](OpenRLHF源码解读&1.理解PPO单机训练_image)

> 图1. PPO训练四阶段

### 1.3.PPO训练四阶段

*   **阶段1：先基于Pretrain model，训练一个**[**精调模型**](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=%E7%B2%BE%E8%B0%83%E6%A8%A1%E5%9E%8B&zhida_source=entity)**（SFT Model）** 和 一个**奖励模型（Reward Model）**。Reward model 一般可以基于SFT model 热启 或 基于 Pretrain model 热启训练
*   **阶段2：模型初始化，PPO过程，在线同时有四个模型，分别为**
    *   **Actor Model ：** 是我们要优化学习的策略模型，同时用于做数据采样，用SFT Model热启
    *   **Reference Model ：** 代码中为initial\_model，是为了控制[Actor模型](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=Actor%E6%A8%A1%E5%9E%8B&zhida_source=entity)学习的分布与原始模型的分布相差不会太远的参考模型，通过loss中增加KL项，来达到这个效果。训练过程中该模型不更新
    *   Critic Model ：是对每个状态做打分的价值模型，衡量当前token到生成结束的整体价值打分，用Reward Model热启
    *   Reward Model ：这里实现的是ORM（Outcome Reward Model），对整个生成的结果打分，是事先训练好的Reward Model。训练过程中该模型不更新
*   阶段3：采样Experience数据，这个过程比较复杂，单独梳理一文。简述流程为：
    *   首先采样一批随机指令集（Prompt）
    *   调用Actor模型的generate()方法，采样1条或多条结果（sequences）
    *   四个模型一起参与组装Experience的多个Tensor域，用于后续模型训练
*   **阶段4:** 用Experience样本，训练 Actor Model 和 Critic Model，后面单独一文介绍

重复3-4阶段，循环采样Experience数据-> 模型训练 ，直到[loss收敛](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=loss%E6%94%B6%E6%95%9B&zhida_source=entity)

上面大体介绍了PPO训练的过程，**下面会继续细化讨论几个关键的问题：**

1.  4个模型结构具体长啥样？Actor Model，Reference Model，Critic Model， Reward Mode
2.  采样过程具体是如何做的？**详见：** [姜富春：OpenRLHF源码解读：2.PPO训练Experience数据采样过程](https://zhuanlan.zhihu.com/p/14569025663)
3.  模型训练过程有哪些细节？**详见：**[姜富春：OpenRLHF源码解读：3.PPO模型训练过程](https://zhuanlan.zhihu.com/p/14813158239)

本文继续讲解下模型结构，采样和模型训练过程已单独拆成两篇文章介绍。

2\. 模型结构
--------

### 2.1. Actor Model 模型结构（Reference Model 同 Actor Model一致）

代码入口 : [Actor Model](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py%23L15)

![](1_OpenRLHF源码解读&1.理解PPO单机训练_image)

> 图2、Actor网络（我们要更新训练的网络）

*   PreTrainModel 和 CausalLM Head 都是Huggingface定义的[标准模型](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=%E6%A0%87%E5%87%86%E6%A8%A1%E5%9E%8B&zhida_source=entity)层。详见：[LlamaForCausalLM类定义](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llama/modeling_llama.py%23L1077C7-L1077C62)
*   2个处理Head：
    *   F.log\_softmax(logits)： 采样经验数据的[数据处理](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86&zhida_source=entity)Head，获取log(p)，方便后面计算KL和计算loss
    *   generate()：采样Head，详见 ：[generate方法定义](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py%23L1907) 。generate可以定义多种生成策略（beam search , sample N等）和配置多种生成参数（topP, temperature等）。详细参数参见：[姜富春：LLM generate方法参数配置-备忘录](https://zhuanlan.zhihu.com/p/14481603550)

### 2.2. Reward Model 模型结构

代码入口：[reward\_model = get\_llm\_for\_sequence\_regression](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo.py%23L58C9-L58C55)

![](2_OpenRLHF源码解读&1.理解PPO单机训练_image)

> 图3、Reward Model网络

*   这里的Reward Model是个ORM（Outcome Reward Model），即对输出的sequence做整体打分，每个输出序列会输出eos位置的打分结果。

### 2.3. Critic Model 模型结构

代码入口： [critic = get\_llm\_for\_sequence\_regression](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo.py%23L39C9-L39C49)

![](3_OpenRLHF源码解读&1.理解PPO单机训练_image)

> 图4、Critic网络（PPO训练阶段要更新的价值评估网络）

*   Critic用于评估当前状态的价值（当前token到生成eos累计预估价值），每个状态都会计算价值打分
*   注： 从图中第二层(Linear层)可以看到，输出结果先做了\[:,:-1\]的切片操作，然后再取生成长度的切片\[:,-num\_actions:\]。这个操作表示整体价值打分序列往前移了一位，这是因为在[生成模型](https://zhida.zhihu.com/search?content_id=251631963&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B&zhida_source=entity)中，一个step数据： (si,ai,si+1,ri) 的描述。当 i=1 时， s1 就是输入的prompt，状态 s1 的end位置是prompt的最后一个token的位置，而这个位置就是上述两次切片操作后的首token位置，表示第一个状态。 a1 是生成的第一个token， s2 是prompt+生成的第一个token， r1 是从 s1→s2 的即时奖励

3.总结
----

本文从OpenRLHF源码梳理了PPO训练过程。通过一张图完整描述了单机PPO训练过程。并对Actor Model, Reward Model， Critic模型做了图示化介绍。

后面会继续通过两篇文章讲解采样过程和模型训练过程。

*   **采样过程：**[姜富春：OpenRLHF源码解读：2.PPO训练Experience数据采样过程](https://zhuanlan.zhihu.com/p/14569025663)
*   **模型训练：**[姜富春：OpenRLHF源码解读：3.PPO模型训练过程](https://zhuanlan.zhihu.com/p/14813158239)