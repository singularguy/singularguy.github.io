# vLLM&V1&代码走读，先睹为快！&1&2&
> _**作者: Uranus**_
> 
> _**原文:**_ [_**https://zhuanlan.zhihu.com/p/14486526122**_](https://zhuanlan.zhihu.com/p/14486526122)

今天看到了 [@游凯超](https://www.zhihu.com/people/176cf88046a1cae595b55e12d58c95e9) 大神分享的 [游凯超：我与vLLM的2024](https://zhuanlan.zhihu.com/p/14430956145?utm_campaign=shareopn&utm_content=group1_article&utm_medium=social&utm_psn=1854945218031398913&utm_source=wechat_session)，感受颇多，[大模型推理](https://zhida.zhihu.com/search?content_id=251920041&content_type=Article&match_order=1&q=%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86&zhida_source=entity)无疑是 2024 年大模型非常重要的一个细分领域，也诞生了非常多优秀的工作，vLLM，sglang，lmdeploy 等开源社区更是其中翘楚。

vLLM 作为大模型推理领域现象级的开源项目，从开源伊始便收获无数 star，形成了庞大且活跃的社区，每天新的 issue 和 pull request 不计其数。

“V1” 的由来
--------

然而随着项目的迅速发展，社区不可避免地开始累积一些 “技术债”，简单来说就是 vLLM 不够 “快” 了。今年 7 月底，sglang 社区一篇略带挑战意味的 [blog](https://link.zhihu.com/?target=https%3A//lmsys.org/blog/2024-07-25-sglang-llama3/) 展示出几乎 3 倍的性能优势（离线场景下）。这让 vLLM 社区开始正视性能问题，并在今年 9 月初通过 v0.6.0 重回开源推理引擎性能的第一梯队。

今年 9 月底，vLLM 社区提出了新架构 “V1”，旨在彻底解决目前的 “[技术债](https://zhida.zhihu.com/search?content_id=251920041&content_type=Article&match_order=2&q=%E6%8A%80%E6%9C%AF%E5%80%BA&zhida_source=entity)”。这是 vLLM 社区目前在积极开发的一个全新引擎架构，可以说是脱胎换骨，截至目前已经初具规模。

“V1” 要解决的很多问题以及解决方案在我司也有一些实践，非常有共鸣，于是想写点什么和大家分享，因此有了这篇文章 :D

“V0” 存在的问题
----------

vLLM 刚开始最为人所知的是 paged attention 和 continuous batching 这两个特性。

它的 continuous batching 实现是在每次 forward 之前，由 scheduler 决定下一个 batch 由哪些 request 组成。这个 batch 将随后被 copy 到 GPU memory 并 broadcast 到各个 GPU，执行 GPU 计算。计算结果通常在 GPU 0 上进行采样，得到本轮迭代生成的 token。而后，这些 token 将会通过 detokenizer，得到对应的字符串。vLLM 将会用这些 token 和字符串更新每一个 request 的状态，为下一次 forward 做好准备。

![](1_vLLM&V1&代码走读，先睹为快！&1&2&_image.)

> “V0” 简化后的推理流程

这个过程中有一些非常耗时的 CPU 操作，比如 scheduling，prepare batch，broadcasting，detokenizing，等等。在这些操作发生时，GPU 是空闲的。这意味着 vLLM “V0” 的 GPU 只有 50%-60% 的时间在工作，其他时间都在围观 CPU 干活。

vLLM 团队也在 “V1” 的[架构文档](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/8779)中总结了这些问题：

![](3_vLLM&V1&代码走读，先睹为快！&1&2&_image.)

> “V1” 架构文档中的总结

简而言之，随着 GPU 越来越快，一切 CPU 操作和通信操作都应该尽可能避免，或者和 GPU 计算 overlap 起来！

“V1” 的新思路
---------

“V1” 在系统设计层面针对性地去解决了上述问题。

### [异步调度](https://zhida.zhihu.com/search?content_id=251920041&content_type=Article&match_order=1&q=%E5%BC%82%E6%AD%A5%E8%B0%83%E5%BA%A6&zhida_source=entity)

对于 scheduling，prepare batch 这两个耗时的操作，“V1” 计划引入一种异步调度机制，在 GPU 正在计算第 N 个 batch 时，提前调度并准备第 N + 1 个 [batch](https://zhida.zhihu.com/search?content_id=251920041&content_type=Article&match_order=8&q=batch&zhida_source=entity)。遗憾的是，目前这个功能还没有合并到主干分支。

### Stateful Worker

“V0” 中，vLLM [推理引擎](https://zhida.zhihu.com/search?content_id=251920041&content_type=Article&match_order=2&q=%E6%8E%A8%E7%90%86%E5%BC%95%E6%93%8E&zhida_source=entity)的所有状态都是在 scheduler 中维护的，这些状态包括每一个请求的 tokens，采样参数，block table 等。在准备 batch 和 broadcast 时，这些状态都需要被 copy 到 GPU 并广播到其他的 GPU 上。

如果我们连续观察几个 batch，我们会发现很大一部分数据是重复的。举例来说，每一个请求的 block table 在相邻的 batch 往往是一样的。因此，为了减少数据传输，我们完全可以把引擎的状态复制到每个 worker 上！

### 独立 Detokenizer 进程

Detokenizer 的作用是将 forward 输出的 token 转化为对应的字符串。而我们知道，LLM 的输入是 token，这些字符串实际上并不是下一轮计算所必须的。因此在 “V1” 中，detokenizer 不再和 engine 处于同一个进程，架构大致变成了这样，新 token 生成后，一方面交给 scheduler 进行下一个迭代，同时也交给 detokenizer 进程解码，将结果发送给用户：

![](2_vLLM&V1&代码走读，先睹为快！&1&2&_image.)

> 独立后的 detokenizer

将上述优化组合起来，“V1” 的推理过程大致如下：

![](vLLM&V1&代码走读，先睹为快！&1&2&_image.)

> “V1” 的推理过程

总结一下，“V1” 是 vLLM 社区为了彻底解决性能问题的一次巨大重构，涉及了 vLLM 的大多数模块。社区仔细分析了 “V0” 存在的诸多问题，并逐一提出了解决对策。在 GPU 越来越快的当下，“V1” 的存在是为了让 GPU 的性能提升最大化地表现在推理效果上。

下一篇文章中，我将分享 “V1” 在代码层面的实现，同时我们会在 timeline 层面分析 “V1” 下的调度，通信，kernel launch，IO 的开销！感兴趣的朋友不妨点个关注 :D

最后，非常感谢 vLLM 团队在[开源](https://zhida.zhihu.com/search?content_id=251920041&content_type=Article&match_order=5&q=%E5%BC%80%E6%BA%90&zhida_source=entity)领域的贡献，这是我最爱的开源社区之一！