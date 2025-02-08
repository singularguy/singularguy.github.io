# [NeurIPS-2023&oral]《QLoRA&Efficient&Finetuning&of&Quantized&LLMs》
> _**作者: 真-忒修斯之船**_  
> _**链接:**_ [_**https://www.zhihu.com/question/630131398/answer/54541623171**_](https://www.zhihu.com/question/630131398/answer/54541623171)

**1.技术解读**
----------

QLoRA(Quantized Low-Rank Adaptation)是一种针对大型预训练语言模型(LLM)的高效微调技术.它结合了量化和低秩适配(LoRA)两种技术,旨在减少模型微调过程中的内存占用和计算成本,同时尽量保持模型性能.

在QLoRA中,首先对模型的权重进行4位量化,这意味着模型的每个权重被表示为4位的数值,显著减少了模型的内存占用.量化后的模型参数以一种称为NormalFloat(NF4)的数据类型存储,这种数据类型特别适合表示正态分布的数据,并且可以比传统的4位整数或浮点数提供更好的量化效果.

接下来,QLoRA利用LoRA技术,通过在模型中引入可训练的[低秩矩阵](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=1&q=%E4%BD%8E%E7%A7%A9%E7%9F%A9%E9%98%B5&zhida_source=entity)来进一步微调模型.这些低秩矩阵作为适配器,被添加到模型的特定层中,并且只有这些适配器的参数在微调过程中被更新,而模型的原始参数保持不变.这样做的好处是,可以针对特定任务微调模型的行为,而不需要对整个模型进行昂贵的更新.

此外,QLoRA还采用了一种称为[双重量化](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=1&q=%E5%8F%8C%E9%87%8D%E9%87%8F%E5%8C%96&zhida_source=entity)的技术,对量化过程中使用的[缩放因子](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=1&q=%E7%BC%A9%E6%94%BE%E5%9B%A0%E5%AD%90&zhida_source=entity)(scale factor)和偏移量([offset](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=1&q=offset&zhida_source=entity))进行再次量化,从而进一步减少内存占用.

QLoRA的另一个关键技术是利用NVIDIA的统一内存进行[分页优化](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=1&q=%E5%88%86%E9%A1%B5%E4%BC%98%E5%8C%96&zhida_source=entity).这种方法可以有效地管理内存使用,特别是在处理长序列数据时,可以避免内存峰值过高的问题.

![]([NeurIPS-2023&oral]《QLoRA&Effi.webp)

**2.直观理解**
----------

想象一下,你有一本厚厚的百科全书,这本书包含了所有的知识(就像一个大型[预训练语言模型](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=2&q=%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)).现在,你想根据你的特定需求来更新这本书(微调模型).但是,整本书重新打印非常耗时且成本高昂.

QLoRA的做法就像是你不需要重新打印整本书,你只需要在书的某些页面上贴几个小便利贴(低秩适配器).这些便利贴包含了你需要的新信息,而且它们很小,不会占用太多空间.

1.  量化: 首先,你把百科全书中的一些文字和图片变成更小的版本,比如把彩色照片换成黑白的,并且缩小尺寸(这就像是4位量化,减少模型大小).
2.  便利贴: 然后,你在需要更新的页面上贴上便利贴,这些便利贴包含了最新的信息(这就像是在模型的关键部分添加低秩适配器).
3.  节省空间: 由于便利贴很小,你不需要为整本书找到额外的空间(这就像是QLoRA减少了模型微调时的内存需求).
4.  保持性能: 尽管你只更新了一小部分内容,但这本书仍然非常有用,因为大部分知识都还在(这就像是QLoRA在减少资源消耗的同时,保持了模型的性能).
5.  更高效: 你不需要重新学习整本书,只需要看看那些便利贴,就可以快速找到你需要的信息(这就像是QLoRA使得[模型微调](https://zhida.zhihu.com/search?content_id=704024498&content_type=Answer&match_order=3&q=%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83&zhida_source=entity)更加高效).

所以,QLoRA就是一种聪明的方法,让你在不重新打印整本百科全书的情况下,快速且经济地更新知识.

**参考**
------

\[1\] [**QLORA: Efficient Finetuning of Quantized LLMs**](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2305.14314)