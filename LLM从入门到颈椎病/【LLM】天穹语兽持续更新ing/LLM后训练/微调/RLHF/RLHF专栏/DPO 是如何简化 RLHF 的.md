# DPO 是如何简化 RLHF 的
> 作者: 
> 
> 原文: 

最近通过 Mistral AI 7Bx8 模型的发布，我才后知后觉地了解到了 [DPO](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=DPO&zhida_source=entity)（Direct Preference Optimization）这个算法，发现他用了一种很巧妙的思路，将 [RLHF](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=RLHF&zhida_source=entity) 的 2 阶段多个模型的训练简化为了 1 阶段的 SFT 训练。在这里简单总结一下。

那么介绍 DPO 做了哪些简化之前，首先要提一下我们一般认为的 RLHF 是咋训练的。RLHF 一般会分 2 步:

*   第一步是训练 [reward model](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)。训练数据是同一个 prompt 的 2 个回答，让人或 [GPT4](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=GPT4&zhida_source=entity) 标注哪个回答更好，reward model 会去优化如下的 loss：  
    maxrϕ{E(x,ywin,ylose)∼D\[log⁡σ(rϕ(x,ywin)−rϕ(x,ylose))\]}\\max\_{r\_{\\phi}}\\left\\{\\mathbb{E}\_{(x,y\_\\text{win},y\_\\text{lose})\\sim\\mathcal{D}}\[\\log\\sigma(r\_\\phi(x,y\_\\text{win})-r\_\\phi(x,y\_\\text{lose}))\]\\right\\}  
    其中 rϕr\_\\phi 就是 reward model 用来给回答打分。D\\mathcal{D} 是训练[数据集](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E9%9B%86&zhida_source=entity)，xx 是 prompt，ywiny\_\\text{win} 和 ylosey\_\\text{lose} 分别是好的回答和不好的回答。也就是说，要尽可能让好的回答的得分比不好的回答高，拉大他们之间的差别。
*   第二步是用 RL 算法来提升模型的得分。使用的 loss 是：  
    maxπθ{Ex∼D,y∼πθ(y|x)\[rϕ(x,y)\]−βDKL\[πθ(y|x)||πref(y|x)\]}\\max\_{\\pi\_\\theta}\\left\\{\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[r\_\\phi(x,y)\]-\\beta\\mathbb{D}\_{\\text{KL}}\[\\pi\_\\theta(y|x)||\\pi\_\\text{ref}(y|x)\]\\right\\}  
    其中 πθ\\pi\_\\theta 是我们在训练的 [LLM](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=LLM&zhida_source=entity)，πref\\pi\_\\text{ref} 是训练的初始值。这个 loss 意思是希望 LLM 输出的回答的评分能尽可能高，同时 πθ\\pi\_\\theta 不要偏离 πref\\pi\_\\text{ref} 太多，保证它还能正常做回答，不要训成一个评分很高但是回答乱码的东西。

DPO 的作者们意识到，后面的这个式子是有显式解的。因为：

maxπθ{Ex∼D,y∼πθ(y|x)\[rϕ(x,y)\]−βDKL\[πθ(y|x)||πref(y|x)\]}=maxπθEx∼D,y∼πθ(y|x)\[rϕ(x,y)−βlog⁡πθ(y|x)πref(y|x)\]=minπθEx∼D,y∼πθ(y|x)\[log⁡πθ(y|x)πref(y|x)−1βrϕ(x,y)\]=minπθEx∼D,y∼πθ(y|x)\[log⁡πθ(y|x)πref(y|x)erϕ(x,y)/β\]\\begin{aligned}\\max\_{\\pi\_\\theta}&\\left\\{\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[r\_\\phi(x,y)\] -\\beta\\mathbb{D}\_{\\text{KL}}\[\\pi\_\\theta(y|x)||\\pi\_\\text{ref}(y|x)\]\\right\\}\\\\&=\\max\_{\\pi\_\\theta}\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[r\_\\phi(x,y) - \\beta \\log \\frac{\\pi\_\\theta(y|x)}{\\pi\_\\text{ref}(y|x)}\]\\\\&=\\min\_{\\pi\_\\theta}\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[\\log \\frac{\\pi\_\\theta(y|x)}{\\pi\_\\text{ref}(y|x)} - \\frac{1}{\\beta} r\_\\phi(x,y)\]\\\\&=\\min\_{\\pi\_\\theta}\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[\\log\\frac{\\pi\_\\theta(y|x)}{\\pi\_\\text{ref}(y|x)e^{r\_\\phi(x,y)/\\beta}}\]\\end{aligned}

如果我们归一化一下分母，即取 Z(x)=∑yπref(y|x)erϕ(x,y)/βZ(x)=\\sum\_y\\pi\_\\text{ref}(y|x)e^{r\_\\phi(x,y)/\\beta}，也就可以构造出一个新的[概率分布](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83&zhida_source=entity)：

π∗(y|x)=πref(y|x)erϕ(x,y)/β/Z(x)\\pi^\*(y|x) = \\pi\_\\text{ref}(y|x)e^{r\_\\phi(x,y)/\\beta}/Z(x)

那么上式变成了：

minπθEx∼D,y∼πθ(y|x)\[log⁡πθ(y|x)πref(y|x)erϕ(x,y)/β\]=minπθEx∼D,y∼πθ(y|x)\[log⁡πθ(y|x)π∗(y|x)−log⁡Z(x)\]=minπθEx∼D,y∼πθ(y|x)\[log⁡πθ(y|x)π∗(y|x)\]=minπθEx∼DDKL(πθ(y|x)||π∗(y|x))\\begin{aligned}\\min\_{\\pi\_\\theta}&\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[\\log\\frac{\\pi\_\\theta(y|x)}{\\pi\_\\text{ref}(y|x)e^{r\_\\phi(x,y)/\\beta}}\]\\\\ &=\\min\_{\\pi\_\\theta}\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[\\log\\frac{\\pi\_\\theta(y|x)}{\\pi^\*(y|x)}-\\log Z(x)\]\\\\ &=\\min\_{\\pi\_\\theta}\\mathbb{E}\_{x\\sim \\mathcal{D},y\\sim\\pi\_\\theta(y|x)}\[\\log\\frac{\\pi\_\\theta(y|x)}{\\pi^\*(y|x)}\]\\\\ &=\\min\_{\\pi\_\\theta}\\mathbb{E}\_{x\\sim \\mathcal{D}}\\mathbb{D}\_\\text{KL}(\\pi\_\\theta(y|x)||\\pi^\*(y|x)) \\end{aligned}

由于 KL [散度](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=%E6%95%A3%E5%BA%A6&zhida_source=entity)在 2 个分布相等时取最小值，我们得到了这样的结论：RLHF 训练希望得到的最优的概率分布就是 \\pi^\*。

另一个角度来说，由 \\pi^\* 的公式，我们相当于是得到了 r\_\\phi 和 \\pi^\* 的关系，那么是否我们可以把训练 r\_\\phi 转化成直接去训练 \\pi^\* 呢？

简单转换一下 \\pi^\* 的定义式，可以得到：

r\_{\\phi}(x,y)=\\beta\\log\\frac{\\pi^\*(y|x)}{\\pi\_\\text{ref}(y|x)}+\\beta \\log Z(x)

带入最上面优化 r\_\\phi 的 loss，也就有了：

\\max\_{\\pi^\*}\\left\\{\\mathbb{E}\_{(x,y\_\\text{win},y\_\\text{lose})\\sim\\mathcal{D}}\[\\log\\sigma(\\beta\\log\\frac{\\pi^\*(y\_\\text{win}|x)}{\\pi\_\\text{ref}(y\_\\text{win}|x)} - \\beta\\log\\frac{\\pi^\*(y\_\\text{lose}|x)}{\\pi\_\\text{ref}(y\_\\text{lose}|x)})\]\\right\\}

或者说，我们可以直接用这个 loss 去求 \\pi\_\\theta：

\\max\_{\\pi\_\\theta}\\left\\{\\mathbb{E}\_{(x,y\_\\text{win},y\_\\text{lose})\\sim\\mathcal{D}}\[\\log\\sigma(\\beta\\log\\frac{\\pi\_\\theta(y\_\\text{win}|x)}{\\pi\_\\text{ref}(y\_\\text{win}|x)} - \\beta\\log\\frac{\\pi\_\\theta(y\_\\text{lose}|x)}{\\pi\_\\text{ref}(y\_\\text{lose}|x)})\]\\right\\}

这就是 DPO 的 loss。DPO 通过以上的[公式转换](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=%E5%85%AC%E5%BC%8F%E8%BD%AC%E6%8D%A2&zhida_source=entity)把 RLHF 无损地转化为了 SFT，在训练的时候不再需要同时跑 4 个模型（reward model, ref model, [critic](https://zhida.zhihu.com/search?content_id=237414431&content_type=Article&match_order=1&q=critic&zhida_source=entity), actor），而是只用跑 actor 和 ref 2 个模型，甚至由于不再在线采数据，ref model 的输出可以预先存下来，训练的时候重复使用。