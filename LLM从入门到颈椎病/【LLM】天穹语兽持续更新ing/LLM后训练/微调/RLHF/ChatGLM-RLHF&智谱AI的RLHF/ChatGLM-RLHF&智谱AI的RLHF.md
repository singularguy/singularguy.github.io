# ChatGLM-RLHF&智谱AI的RLHF
> _**作者: none**_
> 
> _**原文:**_ [_**https://zhuanlan.zhihu.com/p/693059865**_](https://zhuanlan.zhihu.com/p/693059865)

文章：ChatGLM-RLHF: Practices of Aligning Large Language Models with Human Feedback

2024.04 放在arxiv，链接如下：

[https://arxiv.org/pdf/2404.00934.pdfarxiv.org/pdf/2404.00934.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2404.00934.pdf)

### 前言

摘要中提到的ChatGLM-RLHF三个重要的点

*   人类偏好数据的收集
*   reward model的训练
*   policy model的训练

大规模大模型训练中遇到的挑战：

*   减少reward 的方差，使其能在大规模训练中稳定。([biased](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=biased&zhida_source=entity) 和deceptive preferences 在人类标注过程中不可避免)
*   使用融合梯度下降的模型并行(implement model parallelism with fused gradient-descent)
*   设计了正则限制避免灾难性遗忘(design regularization constraints to avoid catastrophic forgetting in LLMs)(更大参数量的模型例如33B以上的，学界很少研究)

### 内容

ChatGLM-RLHF pipeline: 主要聚焦ChatGLM 在中文能力的对齐工作，

*   为了收集reward model的数据：建立一个专门的收集数据系统，使用 ChatGLM SFT model 生成 两个 outputs，让人类根据 helpfulness, harmlessness, and fluency 三个维度打分。
    *   [后处理](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=%E5%90%8E%E5%A4%84%E7%90%86&zhida_source=entity)： 移除了不合理的标注数据:例如：成环的数据[cyclic](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=cyclic&zhida_source=entity) 和 打平的偏好数据。
    *   消除output的长度偏差(such as a biased preference towards longer yet not really helpful outputs),具体怎么消除后面会说

![](ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图1 reward model 数据收集

*   训练PPO 和 DPO

### ChatGLM-RLHF

*   **数据收集和处理：**
    *   要求1：数据要满足多样性(人类的意图/偏好)
    *   数据来源：从现实应用场景收集(估计是ChatGLM开源了，线上用户调用的数据)
    *   处理规则：使用了一个数据质量过滤器选择高质量的数据
        *   质量分类起通过下面三个方面对prompts进行打分：
            *   意图是否明确
            *   语义是否明确
            *   prompt是否可以回答
        *   此外：基于一些规则选择存在信息量的prompts(具体什么规则没说)，文中只是举了一个例子：过滤掉短的prompts。

原文的说明：

![](1_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图2 分类器处理的三个方面

其他： 文章中没有说明 质量分类器怎么训练的，估计就是标注大量类似于图2 的数据，然后使用训练对应的分类器。

一些例子说明：

![](2_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图3 prompts的例子质量例子

```text-plain
- 构建了人类意图多个类别：同时确保每个类别都有足够多的prompts。
- 没有说明sft prompt 数据 和 reward model数据的比例， 是否重复。
```

*   偏好数据的标注
    *   基于sft model 生成两个答案，要求标注人员根据有用性和安全性进行标注哪个答案好。
        *   helpful / safety 的原文说明

![](3_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图4 原文中关于 helpfulness / safety 的说明

```text-plain
- 数据量等的统计：单论/多轮加起来总共221k.
```

![](4_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图5 reward model 数据的训练

*   **Reward Model的训练：**
    *   使用的rank loss 优化reward model

![](5_ChatGLM-RLHF&智谱AI的RLHF_image.j)

```text-plain
- 长度偏差的减少方法(Length Bias Reduction)：
    * 产生的原因：标注员经常优先选择偏长的，结构好的回答(longer, well-structured responses)
    * 解决方案：Bucket-Based Length Balancing(基于[分桶](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=%E5%88%86%E6%A1%B6&zhida_source=entity)的长度平衡)，具体的做法
        + 计算 pair内两个answers的长度差值；提前分好n个桶，每个桶对应一定的差值范围；将对应差值的pair数据放入对应的桶中
```

![](6_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图6 answers 的长度差值

```text-plain
        + 平衡好每一个桶中长response 为chosen和rejected的数据(balance the number of examples where the better/worse response is longer)。这样做无非就是尽量消除掉长度的偏差。
- 来自其他知乎文章关于rm偏差的处理方法，reward model 消除长度偏差的简单粗暴方案。(来自[swtheking：百面LLM-33](https://zhuanlan.zhihu.com/p/690982810))
```

![](7_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图7 rm中的噪声数据处理

```text-plain
- 稳定训练，作者训练的时候发现reward model的值波动比较大，所以引进了L2 正则化的loss (能够变为高斯先验证的0均值分布)。(This loss term imposes a Gaussian prior with a mean of zero on the score distribution, thereby constraining the volatility of the score distribution)，具体loss如下，正负responses的都加
```

![](8_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图8 reg loss 计算

*   **Policy Model Training**
    *   prompt的选择：对一个prompt，多次generate response，如果renspose基本一样，就过滤掉该prompt(感觉generate时候应该适当设调大[温度系数](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=%E6%B8%A9%E5%BA%A6%E7%B3%BB%E6%95%B0&zhida_source=entity)才增加多样性，但文章没说) （Consequently, prompts leading to almost the same responses must be systematically excluded to ensure the effectiveness of this process.）
        *   筛选的方法: 并不是通过对比k个responses的差异，而是通过reward model， 设置一个阈值 ϵ ，如果k个答案的reward方差小于阈值，就过滤掉该prompt。
    *   使用PPO算法。常规的loss如下：

![](9_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图9 RL最大化奖励

![](10_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图10 奖励考虑KL值

```text-plain
- Reward Bias Reduction的其他方法
    * 作者虽然上面使用length bias reduced和L2，但其实未能完全消除；其中一个是数值不稳定(Value instability.),因为还是pairwise training loss，[pairwise loss](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=pairwise+loss&zhida_source=entity) 没有具体的数值限制和回归具体的值，所以[ppo](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=ppo&zhida_source=entity)训练过程中需要reward or advantage normalization 来保持训练的稳定。
    * Task/sample bias：不同任务的prompt 和 response 本来的reward 值分布不一样，不同的tasks 和 prompts, 例如 creative writing, mathematical reasoning, and roleplaying. 作者给出了应该是还没经过ppo训练的reference model在同种task 和不同task是中的reward 分布图：
        + 图中说明，不同task中，rewards 高不一定就是真正的高质量的response，必须考虑 prompt and task的影响。
```

![](11_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图11 Reward distributions on different tasks.

```text-plain
    * 所以作者想出了一个reference baseline 来避免 reward model的绝对数值的不稳定性。具体的做法就是在训练的时候，先对所有的prompt 使用reference model 先generate 出一个答案 yref , 最终的reward 减去这个ref respinse 的reward 值(有点类似于ReMax的做法了)。公式如下：
```

![](12_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图12 减去reference baseline 的reward

```text-plain
    * 所以在早期warm-up training stage, r(x,y) 基本为0，大家可能会纳闷 r(x,y)为0，policy model 和critic model 还能有更新么？其实还是有的。因为critic loss的更新如下， r 为0， critic loss还是不为0的。RLHF的loss 可以参考这篇文章，下图也来自下面这文章：critil model有更新，算advantage自然也不为0，policy model也会更新的，只是前期可能更新得比较慢和稳定而已。
```

[图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读1299 赞同 · 107 评论文章](https://zhuanlan.zhihu.com/p/677607581)

![](13_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图13 critic loss 计算

*   解决灾难遗忘的问题（Capability Forgetting）
    *   作者发现 经过RLHF训练后，模型会遗忘之前的东西，例如自我认识问题/没法要求模型输出json 格式的response。问模型who are you，模型回答："I an AI assistant“(原来SFT的回答是"ChatGLM" )，还是无法正确按照要求回到json 格式的答案。 其实类似的问题[open ai](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=open+ai&zhida_source=entity)也提过，就是对齐税的问题，open ai的instruct GPT 发现RLHF后通用能力会下降，所以增加了pretrain loss。 ChatGLM 在这里的就是加入SFT loss。 没有具体说明SFT loss 的数据prompt + response 与 训练RLHF的 prompt的差距和分布区别和比例。估计就是把之前训练SFT的时候加入吧。 最后的loss 如下：第二项就是sft loss。

![](14_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图14 加入sft loss 防止灾难遗忘

```text-plain
- Direct Preference Optimization (DPO)
    * 作者不是拿训练reward model 的数据来训练DPO，而是依旧训练了一个reward model
        + 估计对拿来训练PPO的prompt 数据生成两个答案，使用这个reward model，来对数据进行chosen 和 rejected 进行标注。作者说分离训练和标注数据训练起来比较稳定(This way separates training data construction and data annotation and is thus more scalable.)(但比较纳闷的一点就是 reward model 本来就存在偏差，再来标注数据，应该是为了跟ppo效果公平比较和节省人力吧，不然使用reward model标注的数据再去训练DPO，偏差更大，肯定还不如让人工直接去标注的。还有就是：reward model 标注的DPO 训练数据的prompt时候跟ppo的是否一样，文章也没说，估计是一样，后面才能公平比较）
        + DPO loss 如下：
```

![](15_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图15 dpo loss

*   **并行训练**
    *   因为作者要全参数训练较大参数的模型(文中展示的32B)，所以在预训练和SFT阶段使用3D[混合并行](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=%E6%B7%B7%E5%90%88%E5%B9%B6%E8%A1%8C&zhida_source=entity)(DP/TP/PP)，所以DPO沿用了这种3D并行。
    *   PPO 因为policy model 需要同时训练和推理生成，pipeline parallel在推理阶段会产生大量的bubbles，会占据80%的时间，所以PPO训练的过程中没有使用PP(估计加大了TP或者DP，文章中提到deepspeed chat 和trlX也使用这样的方法了)

### 实验结果

因为DPO/PPO 都是post-training step after SFT，所以对比的都是原始的SFT Model的效果。有人为评估和自动评估(使用GPT4)

*   训练细节：
    *   PPO训练600-800 iterations,没有使用 reward 和 advantage normalization，因为减去ref baseline。
    *   学习率：1e-6
*   评估数据集： DPO/PPO后的效果一般评估的是在固定问答题目的赢率而不是通用能力数据集的评估。
    *   中文的AlignBench：8大类/ 36 子类任务, 总共683道题。
    *   使用的是greedy strategy 的生成策略。
    *   GPT4 评估的效果如下，打分是1-10分。

![](16_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图16 各个类别题目的分数

```text-plain
- PPO几点结论
    * 与SFT相比：
        + 7B模型 PPO VS SFT 分数： 4.90 VS 4.58
        + 32B模型 PPO VS SFT 分数：6.75 VS 6.37
        + 32B PPO 能超过 deepseek 67B 和got3.5
        + 在Writing, OpenQA, Role-Play等偏主观上显著超过SFT
        + 但是在math and logic 等需要深入思考推理(批判性思维技能)的类型上基本没有提升。可能目前DPO/PPO还不大擅长这一点
        + 额外补充最近（2024.04.19）llama3的一点结论：最近llama3提到DPO/PPO也是对代码和推理题的提升有帮助的。
            - 通过PPO和DPO从偏好排名中学习，也大大提高了[Llama 3](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=Llama+3&zhida_source=entity)在推理和编码任务上的表现。llama3作者发现，如果你向模型提出一个它难以回答的推理问题，模型有时会产生正确的推理轨迹：模型知道如何产生正确的答案，但它不知道如何选择它。在偏好排名上的训练使模型学会了如何选择它。
```

[https://ai.meta.com/blog/meta-llama-3/ai.meta.com/blog/meta-llama-3/](https://link.zhihu.com/?target=https%3A//ai.meta.com/blog/meta-llama-3/)

![](17_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图17 llama3 中提到PPO/DPO

*   PPO稍微好过DPO 0.07分。

**内部**[**数据集**](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=3&q=%E6%95%B0%E6%8D%AE%E9%9B%86&zhida_source=entity)**的Human Evaluation**

*   题目：内部中文400题，与SFT的胜率对比。
*   结果如下：

![](18_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图18 人工评估

*   因为上面自动评估PPO超过了DPO，所有只评估PPO VS SFT，数学/推理推理等较SFT提升不大。文本创作/角色扮演等提升大。

**Reward Model 的评估**

*   在人工构建标注的测试集合，reward model的准确率如下：

![](19_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图19 reward model 正确率

*   结论：
    *   32B的reward model稍微好于6B的模型，模型大带来的提升可能有限。
    *   与[llama2](https://zhida.zhihu.com/search?content_id=242143852&content_type=Article&match_order=1&q=llama2&zhida_source=entity)的结果相当，虽然评估数据都不一样，只能简单对比参考。

**response 的长度对比**

![](20_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图20 长度对比

*   结论：32B或者6B中，ppo的长度比DPO的短，之前ppo评估的分数也高，说明效果提升不全是长度带来的，而是ppo真正提升了回答的质量。
    *   DPO 生成的答案长度更长更明显
    *   PPO也会适当增加回答的长度

有没有使用reference reward baseline 的结果

![](21_ChatGLM-RLHF&智谱AI的RLHF_image.j)

> 图21 w/o reference reward 对比

使用了reference reward 奖励从0开始上涨，从训练曲线上能更稳定点。另外，KL发散得比较慢点。