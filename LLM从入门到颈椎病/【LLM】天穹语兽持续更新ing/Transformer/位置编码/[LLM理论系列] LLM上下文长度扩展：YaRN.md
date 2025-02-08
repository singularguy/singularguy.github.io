# [LLM理论系列] LLM上下文长度扩展：YaRN
LLM上下文扩展系列文章：

[\[LLM理论系列\] RoPE 方法+代码实现](https://zhuanlan.zhihu.com/p/20052942525)

[\[LLM理论系列\] LLM上下文长度扩展：Position Interpolation](https://zhuanlan.zhihu.com/p/20059262902)

[\[LLM理论系列\] NTK-Aware Interpolation](https://zhuanlan.zhihu.com/p/20328774059)

[\[LLM理论系列\] LLM上下文长度扩展：YaRN](https://zhuanlan.zhihu.com/p/20415604232)

在大语言模型(LLM)领域，研究人员一直在探索如何突破上下文窗口长度的限制。本文将介绍其中一篇效果很好的的论文《YARN：EFFICIENT CONTEXT WINDOW EXTENSION OF LARGE LANGUAGE MODELS》提出了一种创新的方法 - YaRN（Yet another RoPE extensioN method）。这种方法不仅在计算效率上表现出色，在模型性能方面也取得了重大突破。

如果希望深入理解YaRN的前置方法，可以阅读我之前写过三篇深入分析上下文窗口扩展技术的文章：

*   [\[LLM理论系列\] RoPE 方法](https://zhuanlan.zhihu.com/p/20052942525)
*   [\[LLM理论系列\] LLM上下文长度扩展：Position Interpolation](https://zhuanlan.zhihu.com/p/20059262902)
*   [\[LLM理论系列\] NTK-Aware Interpolation](https://zhuanlan.zhihu.com/p/20328774059)

**已有方法的不足之处**
-------------

在扩展大语言模型上下文窗口长度的研究中，已有的几种主要方法各有优劣：

*   **RoPE方法**：这是一种创新的位置[编码方案](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E7%BC%96%E7%A0%81%E6%96%B9%E6%A1%88&zhida_source=entity)，通过[复数向量](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E5%A4%8D%E6%95%B0%E5%90%91%E9%87%8F&zhida_source=entity)编码相对位置信息，相比传统的位置编码方法有显著提升。但RoPE的主要局限在于，当[序列长度](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E9%95%BF%E5%BA%A6&zhida_source=entity)超过预训练时的上下文窗口时，其表现会急剧下降，无法有效处理更长的序列。  
     
*   **PI方法**：Position Interpolation通过对RoPE维度进行均匀[插值](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E6%8F%92%E5%80%BC&zhida_source=entity)来扩展上下文窗口。这种方法虽然简单直接，但存在两个明显问题：一是会丢失对长序列处理至关重要的高频信息；二是当扩展倍数 s 较大时，模型性能会显著降低。  
     
*   **NTK-aware Interpolation**：这种方法通过修改RoPE的基底 b′=b⋅Sd/(d−2) 来实现上下文扩展。它在原始预训练模型上效果不错，但在经过微调的模型上表现欠佳。主要原因是该方法在某些低维度上进行了超出预训练范围的外推，这种外推会干扰模型已学到的特征，从而影响微调后的性能。  
     

**YaRN方法**
----------

### **NTK-by-parts插值**

YaRN方法的核心是一种新的插值策略,称为NTK-by-parts插值。为了理解这个方法,我们首先需要引入波长的概念。在RoPE中,对于第 i 个维度,其波长定义为:

\\(\\lambda\_i = \\frac{2\\pi}{\\theta\_i} = \\frac{2\\pi}{b^{-2(i-1)/d}} = 2\\pi b^{2(i-1)/d}\\)

波长 λi 表示在该维度上,位置编码需要经过多少个token才会完成一次完整的旋转周期 (2π) 。

基于波长的概念,YaRN将不同的插值方法分为两类:

*   盲插值方法(blind interpolation): 如PI方法,对所有维度采用相同的插值策略
*   有针对性插值方法: 如YaRN,根据不同维度的特性采用不同的插值策略

为了实现有针对性的插值,YaRN引入了一个重要的比值:

\\(    r(i) = \\frac{L}{\\lambda\_i}\\)

其中 L 是原始上下文长度。这个比值反映了上下文长度与波长的关系。基于这个比值,YaRN构造了一个分段函数:

\\(    \\gamma(r) =     \\begin{cases}          0, & \\text{if } r < \\alpha \\\\         1, & \\text{if } r > \\beta \\\\         \\frac{r-\\alpha}{\\beta-\\alpha}, & \\text{otherwise}     \\end{cases}\\)

最终,YaRN的核心公式为:

这个公式巧妙地实现了对不同频率维度的差异化处理:

1.  低频维度 \\(\\frac{L}{\\lambda\_i} < \\alpha): \\gamma = 0\\), 执行[线性插值](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E6%8F%92%E5%80%BC&zhida_source=entity)\\(h(\\theta\_i)^\\star = \\frac{\\theta\_i}{s}\\)
2.  高频维度 \\(\\frac{L}{\\lambda\_i} > \\beta): \\gamma = 1\\), 保持原始频率 \\(h(\\theta\_i) = \\theta\_i\\)
3.  中频维度 \\(\\alpha \\leq \\frac{L}{\\lambda\_i} \\leq \\beta): \\gamma\\)在0到1之间平滑过渡,实现插值和原始频率的动态混合

以LLaMA模型为例,实验表明 α=1 和 β=32 是较优的参数选择。这意味着:

*   当波长大于上下文长度时,执行线性插值
*   当波长小于上下文长度的1/32时,保持原始频率
*   其他情况下,进行动态混合

### **预softmax缩放机制**

YaRN方法的另一个重要创新是引入预softmax缩放机制。这一机制通过温度参数 t 来动态调节注意力机制，使模型能更好地处理[长序列](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=2&q=%E9%95%BF%E5%BA%8F%E5%88%97&zhida_source=entity)。在将查询向量query和键向量key的点积结果输入到softmax函数之前，将注意力权重的计算公式修改为：

\\(softmax\\left( \\frac{q\_m^T k\_n}{t \\sqrt{|D|}} \\right)\\)

之所以要引入这个机制，是因为当模型的上下文窗口被扩展时，注意力权重的分布会发生显著变化。这种变化会导致模型的困惑度(perplexity)上升，即预测的不确定性增加。为了缓解这一问题，YaRN引入温度参数 t 来对注意力权重进行精细调节：

1.  通过将embedding缩放为原来的 1/t，实现对注意力分布的控制
2.  较小的 t 值会使注意力分布更加"尖锐"，增强模型对长距离依赖的捕捉能力
3.  较大的 t 值则会使注意力分布更加"平滑"，有助于保持模型的稳定性

通过大量实验，研究人员发现温度参数 t 与[扩展因子](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E6%89%A9%E5%B1%95%E5%9B%A0%E5%AD%90&zhida_source=entity) s 之间存在一个优雅的对数关系：

\\(\\frac{1}{t} = 0.1 \\ln(s) + 1\\)

这个公式揭示了一个重要规律：随着我们将上下文窗口扩展得越大（即 s 增大），我们需要相应地降低温度参数 t ，以此来增强模型对长距离信息的敏感度，从而有效降低困惑度，保持模型性能。

**实验结果**
--------

### **训练效率与收敛性**

YaRN在训练效率方面展现出显著优势。实验表明，相比Position Interpolation(PI)方法，YaRN具有更快的收敛速度和更稳定的训练过程。在实验中，将LLaMA 7B模型的上下文窗口扩展至32k时，YaRN仅需400个训练步骤即可达到理想效果，大幅领先于PI方法的训练效率。这种高效的训练特性不仅节省了计算资源，也为模型的实际部署提供了便利。

### **长序列处理能力**

YaRN在处理超长文本序列时表现卓越。通过在Proof-pile和GovReport两个具有代表性的长文本数据集上进行评估，YaRN展示出了出色的长程建模能力。特别值得一提的是，在Proof-pile数据集上，YaRN优化后的LLaMA 2模型能够有效处理128k长度的序列，并取得了2.37的优异困惑度分数，这一成绩显著优于现有其他扩展方法。这证明了YaRN在保持模型性能的同时，成功突破了传统[注意力机制](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=2&q=%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&zhida_source=entity)的长度限制。

### **通用任务表现**

在Hugging Face Open LLM Leaderboard提供的标准化评测中，YaRN扩展后的模型展现出了强大的[泛化能力](https://zhida.zhihu.com/search?content_id=253105324&content_type=Article&match_order=1&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)。实验结果表明，即使在显著扩展上下文窗口的情况下，模型在各类下游任务上的性能仍能与原始模型保持相当。以ARCChallenge为例，YaRN扩展的LLaMA 7B模型达到了52.1%的准确率，与原始模型的53.1%相差无几。这一结果有力地证明了YaRN方法在扩展上下文窗口的同时，能够很好地保持模型的基础能力。

**总结**
------

YaRN方法通过创新性地结合频率域插值和预softmax缩放机制，为大语言模型的上下文窗口扩展提供了一个优雅而高效的解决方案。其主要贡献可以总结为以下几点：

1.  提出了基于频率的自适应插值策略，能够智能地区分和处理不同频率的位置信息
2.  引入了与扩展因子相关的温度参数，有效调节注意力分布，提升长距离建模能力
3.  实现了快速收敛的训练过程，显著降低了计算资源需求
4.  在保持模型基础能力的同时，成功将上下文窗口扩展至128k，并在多个长文本任务上取得了优异成绩

相比其他位置编码扩展方法，YaRN不仅在理论基础上更加扎实，在实际应用中也展现出了更好的性能和更高的实用性。这一方法为大语言模型处理超长文本提供了新的可能，有望在文档分析、长文本理解等领域发挥重要作用。

如果希望深入理解前置方法，如RoPE，PI，NTK-Aware Interpolation方法，欢迎大家看我之前的文章，有更多关于LLM的思考和总结：

*   [\[LLM理论系列\] RoPE 方法](https://zhuanlan.zhihu.com/p/20052942525)
*   [\[LLM理论系列\] LLM上下文长度扩展：Position Interpolation](https://zhuanlan.zhihu.com/p/20059262902)
*   [\[LLM理论系列\] NTK-Aware Interpolation](https://zhuanlan.zhihu.com/p/20328774059)