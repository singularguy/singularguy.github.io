# 如何评价DeepSeek的DeepSeek-V3模型
* * *

created: 2025-01-24T00:20 updated: 2025-01-24T23:59
---------------------------------------------------

> _**作者：Lin Zhang  
> 链接：[https://www.zhihu.com/question/7837132971/answer/65665281923](https://www.zhihu.com/question/7837132971/answer/65665281923)**_

看完技术报告，从infra的视角分享一些个人看法，供大家讨论。

首先，训练超大号的MoE模型，仅使用两千张H800加两个月的时间，就能达到如此好的效果，这点实在是太强了。只能说实践出先知，从DeepSeek过往的技术报告来看，明显可以感觉到团队的算法能力和系统能力都在持续升级。

### 模型结构

遵循system-algorithm co-design原则，DeepSeek-V3继续沿用V2中的MLA和MoE结构，其中前者是为了降低kv cache/token开销，后者是为了降低flops/param开销。

1）**MLA技术**我之前就有介绍[\[1\]](#ref_1)，简单来说就是通过类似[LoRA](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=LoRA&zhida_source=entity)的方式对kv进行降维压缩，同时将升维操作转移到Q和O上，避免反复解压缩。遗憾的是，MLA并没有收获太多关注。一个可能的原因是，它跟MQA相比似乎没有表现出什么优势[\[2\]](#ref_2)，反而增加了系统复杂度。

2）**MoE结构**，不同于[Mixtral](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=Mixtral&zhida_source=entity)中大专家的设计（将稠密模型中的MLP结构复制8份），[DeepSeek-V3](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=2&q=DeepSeek-V3&zhida_source=entity)采用大量“小专家”的设计，能够显著提升模型的稀疏程度（总参数量除以激活参数量）。相比V2的236B总参数（21B激活参数），V3更加激进地引入256个专家，总参数量达到惊人的671B，而激活参数量仅仅增加到37B。

根据技术报告里的数据，得益于更加稀疏的MoE设计，以及系统上的一系列优化，训练V3每trillion数据的GPU小时数仅仅为180K（而V2对应的GPU小时数为172.8K），可谓是将V2技术报告标题中的Economical（性价比）贯彻到底。

3）除了继承V2的模型设计，V3中使用先前发布的**auxiliary-loss-free策略**[\[3\]](#ref_3)来缓解专家之间的负载不均衡（学术探索的技术能够如此迅速地上线到自家大模型，可见DeepSeek对于创新的重视程度）。另外，V3引入了**multi-token prediction（MTP）**，不仅可以在训练时提供更多监督信息，还可以在推理时结合[投机采样](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=%E6%8A%95%E6%9C%BA%E9%87%87%E6%A0%B7&zhida_source=entity)加速模型解码。从论文汇报的效果来看，MTP会是一个不错的训练技巧。

### 训练优化

对于训练而言，最引人注目的自然是**FP8**的使用。DeepSeek-V3据我所知，是第一个（至少在开源社区内）成功使用FP8[混合精度训练](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83&zhida_source=entity)得到的大号MoE模型。

众所周知，FP8伴随着数值溢出的风险，而MoE的训练又非常不稳定，这导致实际大模型训练中BF16仍旧是主流选择。现有FP8方案[\[4\]](#ref_4)的训练困难主要来自两个方面，一个是粗粒度的[per-tensor](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=per-tensor&zhida_source=entity) E4M3量化会因为个别异常值增加量化误差，另一个则是反向过程中使用的E5M2格式会带来较大的舍入误差。

为了解决以上问题，DeepSeek-V3在训练过程中统一使用E4M3格式，并通过**细粒度的per-tile（1x128）和**[**per-group**](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=per-group&zhida_source=entity)**（128x128）量化**来降低误差。这种设计更加接近micro-scaling格式[\[5\]](#ref_5)，然而，当前硬件架构并不支持这种格式的运算，这给FP8矩阵乘法的实现带来了挑战（需要通过partial sum的方式来实现）。

尽管DeepSeek-V3展示了per-tile和per-group量化对于模型收敛的重要性，论文中并没有给出对应的FP8矩阵乘法的算子效率。另外，论文中缺乏per-token加per-channel量化的讨论，不清楚这种实现上更加友好的量化方法对于训练稳定性的影响会有多大。

当然，FP8的好处还体现在**节省显存**上（尤其是激活值）。此外，DeepSeek-V3使用BF16来保存优化器状态，以及对部分操作进行选择性重计算（例如RMSNorm, MLA Up-Proj, SwiGLU）。显存的优化有助于设计更好的并行策略，例如可以减少甚至消除张量并行的使用。

在**并行策略**上，DeepSeek-V3使用64路的专家并行，16路的流水线并行，以及数据并行（ZeRO1）。其中，专家并行会引入[all2all通信](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=all2all%E9%80%9A%E4%BF%A1&zhida_source=entity)，由于每个token会激活8个专家，这导致跨节点的all2all通信开销成为主要的系统瓶颈。

为了降低通信开销，在算法层面，DeepSeek-V3使用**分组路由**的方式，限制每个token只会激活4个节点上的专家，从而减半跨节点的通信流量。在系统层面，将节点间通信和节点内通信进行流水，最大化使用网络带宽和NVLink带宽。

通过以上优化，DeepSeek-V3可以将通信计算比例控制在大约1:1，这为后面的**通信隐藏**带来了机会。具体来说，我们可以将不同[micro-batches](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=micro-batches&zhida_source=entity)里前向和反向的计算通信任务做并发调度，使得计算和通信尽可能相互掩盖。

对于流水线并行，DeepSeek-V3设计了类似于Chimera[\[6\]](#ref_6) 中的双向流水来降低bubble，而没有采用更加常见的interleaved 1F1B（尽管interleaved 1F1B中的steady阶段同样可以将前向和反向的计算通信相互进行隐藏）。

### 推理优化

最后，DeepSeek-V3模型的部署同样十分挑战。

对于MoE模型来说，开源框架大多沿用稠密模型的推理方案，例如Mixtral模型仍旧采用张量并行的方式部署。然而，这种处理方式使得MoE模型相比稠密模型**在推理上失去优势**。这是因为，MoE节省flops的好处主要体现在计算密集的[prefill阶段](https://zhida.zhihu.com/search?content_id=706248170&content_type=Answer&match_order=1&q=prefill%E9%98%B6%E6%AE%B5&zhida_source=entity)，而在访存密集的decode阶段，MoE巨大的参数量然而会带来更加昂贵的数据搬移开销。哪怕能解决访存密集的问题，MoE参数消耗如此多昂贵的HBM空间，这可能也不是一个相当划算的决定。

可见，要发挥出MoE架构在推理侧的价值，必须改变并行策略，回到训练时**DP+EP**的方式。这意味着我们需要使用更大的机器单元来部署MoE模型，并尽可能避免专家层的冗余存储，从而降低每个设备上的模型参数量，缓解HBM容量和带宽的压力。

在这种部署方案下，负载均衡和all2all通信成为了核心挑战。了解以上背景之后，让我们回到DeepSeek-V3的推理方案。

首先，DeepSeek-V3采取**PD分离**的方式，分别应对prefill和decode两阶段的挑战。

在**prefill阶段**，attention模块采用4路张量并行+8路数据并行，moe模块采用32路专家并行。这样并行的目的是在满足首token时延的要求下，最大化系统吞吐（和训练任务类似）。

在**decode阶段**，DeepSeek-V3采取320路专家并行（256个小专家+64个热点专家），有效降低解码时延，并缓解负载不均衡的问题。

最后，为了填充all2all通信阶段的设备空闲时间，DeepSeek-V3采用NanoFlow[\[7\]](#ref_7)中的双流推理策略，将不同micro-batch中的计算和通信任务并发执行，从而提高设备资源利用率。

参考
--

1.  [^](#ref_1_0)如何看待 DeepSeek 发布的 MoE 大模型 DeepSeek-V2？ [https://zhihu.com/question/655172528/answer/3504750755](https://zhihu.com/question/655172528/answer/3504750755)
2.  [^](#ref_2_0)MLA通过增加attention head数量来弥补精度损失，同样的技巧也可以应用到MQA。目前缺少二者的公平对比
3.  [^](#ref_3_0)Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts， [https://arxiv.org/abs/2408.15664](https://arxiv.org/abs/2408.15664)
4.  [^](#ref_4_0)Using FP8 with Transformer Engine， [https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8\_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
5.  [^](#ref_5_0)Microscaling Data Formats for Deep Learning, [https://arxiv.org/abs/2310.10537](https://arxiv.org/abs/2310.10537)
6.  [^](#ref_6_0)Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines， [https://arxiv.org/abs/2107.06925](https://arxiv.org/abs/2107.06925)
7.  [^](#ref_7_0)NanoFlow: Towards Optimal Large Language Model Serving Throughput， [https://arxiv.org/html/2408.12757v1](https://arxiv.org/html/2408.12757v1)