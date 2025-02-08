# Transformer 模型是如何想到的，背后的思路和启发点有哪些，有哪些历史渊源，有相关问题吗？
> _**作者：太阳面包**_  
> _**链接：**_[_**https://www.zhihu.com/question/10099752399/answer/84392430400**_](https://www.zhihu.com/question/10099752399/answer/84392430400)

最早是attention，seq2seq模型里面的decoder的每一步都可以用attention查询encoder在不同时间步的状态，可以缓解隐状态有限导致的信息丢失问题。

之后[bengio](https://zhida.zhihu.com/search?content_id=709993255&content_type=Answer&match_order=1&q=bengio&zhida_source=entity)组的[一篇论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.03130)为了改善句子嵌入的信息丢失问题，利用当时还比较流行的加性注意力机制设计了一种从bilstm的隐藏步状态中提取信息的注意力机制，因为这种注意力将查询、键和值合三为一了，论文就将其叫做自注意力。要注意的是，这篇论文提出的自注意力和我们现在所说的自注意力机制长相非常不一样，它基于[加性注意力](https://zhida.zhihu.com/search?content_id=709993255&content_type=Answer&match_order=2&q=%E5%8A%A0%E6%80%A7%E6%B3%A8%E6%84%8F%E5%8A%9B&zhida_source=entity)，而且输出的注意力矩阵尺寸是固定的，不随输入序列的长度变化而改变。

google brain的团队想要扬掉阻碍并行的rnn架构（lstm和[gru](https://zhida.zhihu.com/search?content_id=709993255&content_type=Answer&match_order=1&q=gru&zhida_source=entity)）。作者们觉得新出的自注意力（注意这里说的是bengio那个自注意力）是一个比较好的替代方案，因为[注意力机制](https://zhida.zhihu.com/search?content_id=709993255&content_type=Answer&match_order=4&q=%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&zhida_source=entity)在rnn领域得到广泛使用，效果也不差，其中一位作者之前就用注意力做过[类似的尝试](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1606.01933)。结果作者团队试了几个选型，要么模型不能work，要么需要堆大量trick才能work。后来作者团队天降猛人shazeer，这哥们通过大量消融把自注意力改成了今天的样子（多头缩放点积自注意力，输入和输出尺寸一致），于是attention is all you need横空出世。

至于为什么叫transformer，一个原因是它的架构从一开始就接近于对输入序列的自注意力表征反复做变换，但更重要的原因恐怕是作者们喜欢变形金刚。[\[1\]](#ref_1)

* * *

ps：8年过去了，从现在来看猛男shazeer设计的自注意力选型依旧毫不过时。交叉注意力扬了，ffn变glu了，位置编码改了，残差位置变了，甚至layernorm的均值操作都无了，唯独这个多头缩放点积自注意力依然坚挺。。。[\[2\]](#ref_2)

参考
--

1.  [^](#ref_1_0)本文中的许多推测均来自于wired杂志对transformer作者团队的采访. [https://www.wired.com/story/eight-google-employees-invented-modern-ai-transformers-paper/](https://www.wired.com/story/eight-google-employees-invented-modern-ai-transformers-paper/)
2.  [^](#ref_2_0)严格说来shazeer老哥也动过多头自注意力结构，改成了MQA，但这种改进主要是为了提升KV Cache性能（decoder-only专属），而且很快被GQA取代了，这个更接近原始的MHA一些.

*   ffn到glu也是shazeer设计的 [https://arxiv.org/pdf/2002.05202](https://arxiv.org/pdf/2002.05202)
*   多个Query共享一组Key 和Value也是shazeer设计的 [https://arxiv.org/pdf/1911.02150](https://arxiv.org/pdf/1911.02150)