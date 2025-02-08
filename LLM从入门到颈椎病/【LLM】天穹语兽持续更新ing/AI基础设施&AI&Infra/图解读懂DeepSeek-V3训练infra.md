# 图解读懂DeepSeek-V3训练infra
* * *

created: 2025-01-24T23:58 updated: 2025-01-25T00:00
---------------------------------------------------

> _**作者: 虚怀若谷​**_ _**原文: [https://zhuanlan.zhihu.com/p/20135994467](https://zhuanlan.zhihu.com/p/20135994467)**_

DeepSeek-V3成功的[infra](https://zhida.zhihu.com/search?content_id=253049485&content_type=Article&match_order=1&q=infra&zhida_source=entity)
-----------------------------------------------------------------------------------------------------------------------------------------

开诚布公的介绍了如何高效构建这个模型训练系统，对于很多国内资源少的团队其实帮助很大。

我推荐读读代码框架里给的inference部分，初学者可以看看hf兼容的modeling代码。官方的核心操作FP8算子都开源了；至于基于SMs 的Compute+Communicate-chunks 调度最优tuning实现DualPipe，这个目前没开源，还是需要各个机构 根据自己的集群进行tuning建设。

[DeepSeek-V3 · 模型库](https://link.zhihu.com/?target=https%3A//modelscope.cn/models/deepseek-ai/DeepSeek-V3/files); [GitHub - deepseek-ai/DeepSeek-V3](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepSeek-V3)

![](7_图解读懂DeepSeek-V3训练infra_image.j)

传统大规模下并行训练
----------

为了大家更好理解，这里先做一点简单的传统meagtron类大规模并行训练背景介绍。我觉得meta之类肯定类似，因为他们用了很大的计算资源，只有6-D并行才能维持消耗（tensor、[pipeline](https://zhida.zhihu.com/search?content_id=253049485&content_type=Article&match_order=1&q=pipeline&zhida_source=entity)、sequence、context、data、expert(只针对MOE，meta不算，mixtral算)）。

![](9_图解读懂DeepSeek-V3训练infra_image.j)

传统模型训练框架并行方案，例如megatron

![](5_图解读懂DeepSeek-V3训练infra_image.j)

TP/CP/SP

![](16_图解读懂DeepSeek-V3训练infra_image.j)

PP优化的4个level

![](图解读懂DeepSeek-V3训练infra_image.j)

PP的理论极限优化

### DeepSeek在HAI-LLM上做出的V3

![](8_图解读懂DeepSeek-V3训练infra_image.j)

首先分析下DualPipe

![](13_图解读懂DeepSeek-V3训练infra_image.j)

![](4_图解读懂DeepSeek-V3训练infra_image.j)

可以看到，实现DualPipe里面的计算和通信编排，需要对GPU的流处理做底层调度，这里穿插一个GPU cuda模型的说明

![](12_图解读懂DeepSeek-V3训练infra_image.j)

下面时deepSeek-V3做的D/C算子编排优化，主要用在MoE上

![](15_图解读懂DeepSeek-V3训练infra_image.j)

![](11_图解读懂DeepSeek-V3训练infra_image.j)

![](2_图解读懂DeepSeek-V3训练infra_image.j)

### FP8优化原理【GPTQ启发】

此外，DeepSeek还有一个杀手锏，就是FP8训练，其实核心就是计算乘法部分用FP8，其他数据部分用BF16/FP32.具体如下

![](1_图解读懂DeepSeek-V3训练infra_image.j)

不要被FP8计算吓到，他们其实用到的核心跟量化操作区别不大，不过做了scale FP32<->fp8变换而已。看过GPTQ会很好理解这种block-wise scale/rescale技术方案。

![](3_图解读懂DeepSeek-V3训练infra_image.j)

最后说一点推理和硬件

![](6_图解读懂DeepSeek-V3训练infra_image.j)

![](14_图解读懂DeepSeek-V3训练infra_image.j)

理想还是很丰满的，就是不知道在HBM里支持这些操作对于Nvidia来说有没有难度，鄙人认为要做些可能会在HBM上做一个类似L3-cache区，直接访存操作。

文末，附上DeepSeek团队对于未来发展的规划

![](10_图解读懂DeepSeek-V3训练infra_image.j)