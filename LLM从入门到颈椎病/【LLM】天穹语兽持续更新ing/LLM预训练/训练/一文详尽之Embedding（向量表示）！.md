# 一文详尽之Embedding（向量表示）！
> _**作者: DataWhale**_
> 
> _**原文:**_ [_**https://mp.weixin.qq.com/s/vNO6o3R9SM0C\_\_cZnNoNvw**_](https://mp.weixin.qq.com/s/vNO6o3R9SM0C__cZnNoNvw)

一文详尽之Embedding(向量表示)！
---------------------

在大模型时代,Embedding(向量表示)的重要性愈发凸显.

不论是在 RAG 系统,还是在跨模态任务中都扮演着关键角色.

本文带你详细了解文本向量的前世今生,让你能够更好地了解向量,使用向量.

**1\. 文本表示模型简介**
----------------

文本是一类非常重要的非结构化数据,如何表示文本数据一直是机器学习领域的一个重点研究方向.文本向量就是深度学习时代产生的一种文本表示的方法.

**1.1 独热编码**

最简单的文本表示模型就是独热编码(One-Hot Encoding),用于将词汇表中的每个词转换为一个高维稀疏向量,每个词的向量中只有一个位置为1,其余位置为0.假设我们有一个词汇表(词典)V,其中包含N个唯一的词.每个词可以表示为一个长度为N的二进制向量.在这个向量中,只有一个位置为1,对应于该词在词汇表中的位置,其余位置均为0.例如,假设词汇表为: \["I", "love", "machine", "learning"\]: "I" 的独热表示为\[1, 0, 0, 0\] "love" 的独热表示为\[0, 1, 0, 0\] "machine" 的独热表示为\[0, 0, 1, 0\] "learning" 的独热表示为\[0, 0, 0, 1\]

这种表示方法的**缺点**:

1.  **\>高维稀疏性**: 对于大词汇表,独热向量的维度非常高且稀疏,导致计算和存储效率低.
2.  **\>无法捕捉词语之间的关系**: 独热表示无法捕捉词语之间的语义关系,所有词被认为是独立且等距的.

**1.2 词袋模型与N-gram模型**
---------------------

在**词袋模型(Bag-of-Words, BoW)** 中,每个文档被表示为一个固定长度的向量.向量的每个位置对应于词汇表中的一个词,向量的值为该词在文档中出现的次数(或其他统计量,如词频-逆文档频率(TF-IDF)).假设有两个文档,文档1: "I love machine learning."和文档2: "I love machine learning. Machine learning is great.":

*   构建词汇表: \["I", "love", "machine", "learning", "is", "great"\]
*   文档1的词袋向量: \[1, 1, 1, 1, 0, 0\]
*   文档2的词袋向量: \[1, 1, 2, 2, 1, 1\]

词袋模型没有考虑词的顺序和上下文信息,**N-gram模型**通过考虑文本中连续 n 个词的序列来捕捉词的顺序和局部上下文信息.n-gram 可以是 unigram(1-gram),bigram(2-gram),trigram(3-gram)等.假设有一个文档: "I love machine learning.":

*   生成 bigram(2-gram): \["I love", "love machine", "machine learning"\]
*   构建词汇表: \["I love", "love machine", "machine learning"\]
*   文档的 bigram 向量: \[1, 1, 1\]

**1.3 主题模型**
------------

基于词袋模型与N-gram模型的文本表示模型有一个明显的缺陷,就是**无法识别出两个不同的词或词组具有相同的主题**.因此,需要一种技术能够将具有相同主题的词或词组映射到同一维度上去,于是产生了主题模型.**主题模型**(代表pLSA模型,LDA模型)是一种特殊的概率图模型,主要解决了两个问题:

*   从文本库中发现有代表性的主题(主题数作为超参数需要自己定义)->词分布,表示主题和词的对应关系
*   每篇文章对应着哪些主题->主题分布,表示文档和主题的对应关系

**主题模型的优点**:

*   全局主题建模,能够发现长距离词依赖信息；
*   将文档表示为主题的概率分布,通常维度较低(等于主题数),可以有效地降低数据的稀疏性.贝叶斯学派和频率学派“正统”之争([https://www.zhihu.com/question/20587681/answer/23060072](https://www.zhihu.com/question/20587681/answer/23060072))

**1.4 词向量**
-----------

**词向量(词嵌入,Word Embedding)** 是将每个词都映射到低维空间上的一个稠密向量(Dense Vector).这里的低维空间上的每一维也可以看作是中一个主题模型中的一个主题,只不过不像主题模型中那么直观(不具备可解释性).词向量在学习时考虑到了当前词的上下文信息,以Word2Vec为例,它实际上是一种浅层的神经网络模型(输入层、映射层、输出层),它有两种网络结构,分别是CBOW和Skip-gram,参考下图:

![](一文详尽之Embedding（向量表示）！_image.jp)

不管是CBOW还是Skip-gram,都是通过**最大化上下文词出现的概率来进行模型的优化**.其他常见的词嵌入方法,还有**GloVe(Global Vectors for Word Representation)、FastText**.

**词向量的几何性: 词嵌入的向量空间具备几何性**,因此允许使用**向量之间的距离和角度来衡量相似性**.常用的相似度度量包括:

*   **\>余弦相似度**: 衡量两个向量之间的角度,相似度越高,角度越小.
*   **\>欧氏距离**: 衡量两个向量之间的直线距离,距离越小,词语之间越相似.

![](15_一文详尽之Embedding（向量表示）！_image.we)

词向量存在的问题: 

1.  \>上下文无关:  词向量为静态向量,即同一个词在不同的上下文中有相同的向量表示,这导致词向量无法解决“一词多义”问题.
2.  \>OOV问题:  未登录词问题(Out-Of-Vocabulary),词嵌入模型在训练时建立一个固定的词汇表,任何不在词汇表中的词都被视为 OOV.
3.  \>语法和语义信息的捕捉不足:  词嵌入模型主要捕捉词语的共现信息,可能无法充分捕捉词语之间的复杂语法和语义关系.

**1.5 句向量**
-----------

**句向量(Sentence Embedding)** 是将整个句子转换为固定长度的向量表示的方法.最简单的句向量获取方式是基于平均词向量的方法: 将句子中的每个词转换为词向量,然后对这些词向量取平均得到句子向量.

### **1.5.1 词向量+sif或者usif**

上面介绍的词向量相加求平均获取句向量的方法,缺点之一是没有考虑到句子中不同词之间的重要性差异,因为所有词都是一样的权重.在实际使用过程中,有尝试使用tf-idf对词进行加权,但是效果提升有限.2016年提出来SIF加权算法,使用随机游走来推算句子的生成概率.2018年发表在ACL的USIF(「Unsupervised Random Walk Sentence Embeddings」[https://aclanthology.org/W18-3012.pdf)算法,是对SIF加权算法的改进.感兴趣的可以看看「这一篇」https://zhuanlan.zhihu.com/p/263511812](https://aclanthology.org/W18-3012.pdf)%E7%AE%97%E6%B3%95,%E6%98%AF%E5%AF%B9SIF%E5%8A%A0%E6%9D%83%E7%AE%97%E6%B3%95%E7%9A%84%E6%94%B9%E8%BF%9B.%E6%84%9F%E5%85%B4%E8%B6%A3%E7%9A%84%E5%8F%AF%E4%BB%A5%E7%9C%8B%E7%9C%8B%E3%80%8C%E8%BF%99%E4%B8%80%E7%AF%87%E3%80%8Dhttps://zhuanlan.zhihu.com/p/263511812).

### **1.5.2 Doc2vec**

Doc2vec是基于 Word2Vec 模型扩展而来,相对于word2vec不同之处在于,在输入层,增添了一个新的句子向量**Paragraph vector**,Paragraph vector可以被看作是另一个词向量.Doc2vec每次训练也是**滑动截取句子中一小部分词来训练**,Paragraph Vector在**同一个句子的若干次训练中是共享的**,这样训练出来的Paragraph vector就会逐渐在每句子中的几次训练中不断稳定下来,形成该句子的主旨.

类似于 Word2Vec,它也包含两种训练方式:

1.  **PV-DM (Distributed Memory Model of paragraph vectors)模型**
    *   给定一个文档D和一个窗口大小\_>w\_,选择一个目标词\_>w\_t\_和它的上下文词\_>w\_{t-1}, ..., w\_{t+w}\_
    *   将文档D的向量和上下文词的向量一起输入模型,进行相加求平均或者累加构成一个新的向量\_>X\_,基于该向量预测目标词\_>w\_t\_

![](16_一文详尽之Embedding（向量表示）！_image.we)

1.  **PV-DBOW(Distributed Bag of Words of paragraph vector)模型**
    *   给定一个文档D,随机选择一个目标词\_>w\_t\_
    *   使用文档D的向量预测目标词\_>w\_t\_

![](1_一文详尽之Embedding（向量表示）！_image.we)

**Doc2vec的推理优化: 在预测新的\*句子向量(推理)时,Paragraph vector首先被随机初始化**,放入模型中再重新根据**随机梯度下降**不断迭代求得最终稳定下来的句子向量.与训练时相比,此时**该模型的词向量和投影层到输出层的soft weights参数固定**,只剩下Paragraph vector用梯度下降法求得,所以预测新句子时虽然也要放入模型中不断迭代求出,**相比于训练时,速度会快得多**.

**2\. 基于Bert的文本向量**
-------------------

**2.1 Bert向量的各向异性**
-------------------

将文本输入Bert模型中后,通常使用最后一个隐藏层中\[CLS\]标记对应的向量或对最后一个隐藏层做MeanPooling,来获取该文本对应的向量表示.

![](3_一文详尽之Embedding（向量表示）！_image.we)

为什么Bert向量表示不能直接用于相似度计算？核心原因就是: **各向异性(anisotropy)**.各向异性是一个物理学、材料科学领域的概念,原本指材料或系统在不同方向上表现出不同物理性质的现象.作为吃货的我们肯定非常有经验,吃肉的时候顺纹切的肉比横纹切的肉更有嚼劲.类似的还有木头的顺纹和横纹的抗压和抗拉能力也不同,石墨单晶的电导率在不同方向差异很大.**各向异性在向量空间上的含义就是分布与方向有关系,而各向同性就是各个方向都一样**,比如二维的空间,各向异性和各向同性对比如下(左图为各向异性,右图为各向同性).

![](2_一文详尽之Embedding（向量表示）！_image.we)

**Bert的各向异性是指**: “Anisotropic” means word embeddings occupy a narrow cone in the vector space. 简单解释就是,**词向量在向量空间中占据了一个狭窄的圆锥形体**.Bert的词向量在空间中的分布呈现锥形,研究者发现高频的词都靠近原点(所有词的均值),而低频词远离原点,这会导致即使一个高频词与一个低频词的语义等价,但是词频的差异也会带来巨大的差异,从而词向量的距离不能很好的表达词间的语义相关性.另外,还存在**高频词分布紧凑,低频词的分布稀疏**的问题.分布稀疏会导致区间的语义不完整(poorly defined),低频词向量训练的不充分,从而导致进行向量相似度计算时出现问题.

![](一文详尽之Embedding（向量表示）！_image.we)

不只是Bert,大多数Transformer架构的预训练模型学习到的词向量空间(包括GPT-2),都有类似的问题(感兴趣的可以去看「Gao et al. 2019」[https://arxiv.org/abs/1907.12009,「Wang](https://arxiv.org/abs/1907.12009,%E3%80%8CWang) et al. 2020」[https://openreview.net/pdf?id=ByxY8CNtvr,「Ethayarajh](https://openreview.net/pdf?id=ByxY8CNtvr,%E3%80%8CEthayarajh), 2019」[https://arxiv.org/abs/1909.00512](https://arxiv.org/abs/1909.00512)):

![](6_一文详尽之Embedding（向量表示）！_image.we)

**各项异性的缺点:  一个好的向量表示应该同时满足**alignment**和**uniformity,前者表示**相似的向量距离应该相近**,后者就表示**向量在空间上应该尽量均匀**,最好是各向同性的.各向异性就有个问题,那就是最后学到的**向量都挤在一起,彼此之间计算余弦相似度都很高**,并不是一个很好的表示.

![](4_一文详尽之Embedding（向量表示）！_image.we)

各项异性问题的优化方法: 

1.  **\>有监督学习优化:  通过标注语料构建双塔Bert或者单塔Bert来进行**\>模型微调,使靠近下游任务的Bert层向量更加靠近句子相似embedding的表达,从而使向量空间平滑.有监督学习优化的代表是>SBERT.
2.  \>无监督学习优化: 
    1.  通过对Bert的向量空间进行>线性变换,缓解各向异性的问题.无监督学习优化的代表是>Bert-flow**和**\>Bert-whitening.
    2.  基于>对比学习**的思想,通过对文本本身进行**\>增广(转译、删除、插入、调换顺序、Dropout等),扩展出的句子作为正样本,其他句子的增广作为负样本,通过拉近正样本对的向量距离,同时增加与负样本的向量距离,实现模型优化,又叫>自监督学习优化,代表就是>SimCSE.

**2.2 有监督学习优化-Sentence-BERT(SBERT)**
------------------------------------

BERT向量模型(SBERT模型)与BERT交互式语义模型(BERT二分类模型)的区别:

![](8_一文详尽之Embedding（向量表示）！_image.we)

SBERT模型通过对预训练的BERT结构进行修改,并通过在有监督任务(如自然语言推理、语义文本相似度)上进行微调,优化Bert向量直接进行相似度计算的性能.

![](17_一文详尽之Embedding（向量表示）！_image.we)

SBERT模型分为**孪生(Siamese)** 和**三级(triplet)** 网络,针对不同的网络结构,设置了三个不同的目标函数:

1.  **\>Classification Objective Function**: 孪生(Siamese)网络训练,针对分类问题.上左图,\_>u,\_\_>v\_分别表示输入的2个句子的向量表示,\*|u-v|\*表示取两个向量element-wise的绝对值.模型训练时使用交叉熵损失函数计算loss.

![](5_一文详尽之Embedding（向量表示）！_image.we)

1.  **\>Regression Objective Function**: 孪生(Siamese)网络训练,针对回归问题.上右图,直接计算u,v向量的余弦相似度,训练时使用均方误差(Mean Squared Error, MSE)进行loss计算.3.**\>Triplet Objective Function**: 三级(triplet)网络训练,模型有变成三个句子输入.给定一个锚定句\_>𝑎\_,一个肯定句\_>𝑝_和一个否定句_\>𝑛\_,模型通过使\_>𝑎_与_\>𝑝_的距离小于与_\>𝑎_与_\>𝑛\_的距离,来优化模型,即

![](7_一文详尽之Embedding（向量表示）！_image.we)

其中,_𝜖_ 表示边距,在论文中,距离度量为欧式距离,边距大小为1.

同一时间的其他针对句向量的研究,「USE(Universal Sentence Encoder)」[https://arxiv.org/abs/1803.11175、「InferSent」https://arxiv.org/abs/1705.02364等,他们的共性在于都使用了“双塔”类的结构](https://arxiv.org/abs/1803.11175%E3%80%81%E3%80%8CInferSent%E3%80%8Dhttps://arxiv.org/abs/1705.02364%E7%AD%89,%E4%BB%96%E4%BB%AC%E7%9A%84%E5%85%B1%E6%80%A7%E5%9C%A8%E4%BA%8E%E9%83%BD%E4%BD%BF%E7%94%A8%E4%BA%86%E2%80%9C%E5%8F%8C%E5%A1%94%E2%80%9D%E7%B1%BB%E7%9A%84%E7%BB%93%E6%9E%84).

关于自然推理任务？自然语言推理(Natural Language Inference, NLI)是自然语言处理(NLP)中的一个重要任务,旨在判断两个给定句子之间的逻辑关系.具体来说,NLI任务通常涉及三个主要的逻辑关系:

1.  **\>蕴含(Entailment)**: 前提(Premise)蕴含假设(Hypothesis),即如果前提为真,则假设也为真.
2.  **\>矛盾(Contradiction)**: 前提与假设相矛盾,即如果前提为真,则假设为假.
3.  **\>中立(Neutral)**: 前提与假设既不蕴含也不矛盾,即前提为真时,假设可能为真也可能为假.

**使用NLI的标注数据集(SNLI,MNLI等)训练出的语义向量模型,展现出了较好的语义扩展性和鲁棒性,适合进行zero-shot或者few-shot.**

**2.3 无监督学习优化**
---------------

### **2.3.1 Bert-flow**

既然Bert向量空间不均匀,那就想办法把该向量空降映射一个均匀的空间中,例如**标准高斯(Standard Gaussian)** 分布.「BERT-flow」[https://arxiv.org/abs/2011.05864,就是基于流式生成模型,将BERT的表示可逆地映射到一个标准高斯分布上.具体做法是,在Bert后面接一个flow模型,flow模型的训练是无监督的,并且Bert的参数是不参与训练,只有flow的参数被优化,而训练的目标则为最大化预先计算好的Bert的句向量的似然函数.训练过程简单说就是学习一个可逆的映射𝑓,把服从高斯分布的变量𝑧映射到BERT编码的𝑢,那𝑓−1就可以把𝑢映射到均匀的高斯分布,这时我们最大化从高斯分布中产生BERT表示的概率,就学习到了这个映射](https://arxiv.org/abs/2011.05864,%E5%B0%B1%E6%98%AF%E5%9F%BA%E4%BA%8E%E6%B5%81%E5%BC%8F%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B,%E5%B0%86BERT%E7%9A%84%E8%A1%A8%E7%A4%BA%E5%8F%AF%E9%80%86%E5%9C%B0%E6%98%A0%E5%B0%84%E5%88%B0%E4%B8%80%E4%B8%AA%E6%A0%87%E5%87%86%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E4%B8%8A.%E5%85%B7%E4%BD%93%E5%81%9A%E6%B3%95%E6%98%AF,%E5%9C%A8Bert%E5%90%8E%E9%9D%A2%E6%8E%A5%E4%B8%80%E4%B8%AAflow%E6%A8%A1%E5%9E%8B,flow%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83%E6%98%AF%E6%97%A0%E7%9B%91%E7%9D%A3%E7%9A%84,%E5%B9%B6%E4%B8%94Bert%E7%9A%84%E5%8F%82%E6%95%B0%E6%98%AF%E4%B8%8D%E5%8F%82%E4%B8%8E%E8%AE%AD%E7%BB%83,%E5%8F%AA%E6%9C%89flow%E7%9A%84%E5%8F%82%E6%95%B0%E8%A2%AB%E4%BC%98%E5%8C%96,%E8%80%8C%E8%AE%AD%E7%BB%83%E7%9A%84%E7%9B%AE%E6%A0%87%E5%88%99%E4%B8%BA%E6%9C%80%E5%A4%A7%E5%8C%96%E9%A2%84%E5%85%88%E8%AE%A1%E7%AE%97%E5%A5%BD%E7%9A%84Bert%E7%9A%84%E5%8F%A5%E5%90%91%E9%87%8F%E7%9A%84%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0.%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E7%AE%80%E5%8D%95%E8%AF%B4%E5%B0%B1%E6%98%AF%E5%AD%A6%E4%B9%A0%E4%B8%80%E4%B8%AA%E5%8F%AF%E9%80%86%E7%9A%84%E6%98%A0%E5%B0%84%F0%9D%91%93,%E6%8A%8A%E6%9C%8D%E4%BB%8E%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E7%9A%84%E5%8F%98%E9%87%8F%F0%9D%91%A7%E6%98%A0%E5%B0%84%E5%88%B0BERT%E7%BC%96%E7%A0%81%E7%9A%84%F0%9D%91%A2,%E9%82%A3%F0%9D%91%93%E2%88%921%E5%B0%B1%E5%8F%AF%E4%BB%A5%E6%8A%8A%F0%9D%91%A2%E6%98%A0%E5%B0%84%E5%88%B0%E5%9D%87%E5%8C%80%E7%9A%84%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83,%E8%BF%99%E6%97%B6%E6%88%91%E4%BB%AC%E6%9C%80%E5%A4%A7%E5%8C%96%E4%BB%8E%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E4%B8%AD%E4%BA%A7%E7%94%9FBERT%E8%A1%A8%E7%A4%BA%E7%9A%84%E6%A6%82%E7%8E%87,%E5%B0%B1%E5%AD%A6%E4%B9%A0%E5%88%B0%E4%BA%86%E8%BF%99%E4%B8%AA%E6%98%A0%E5%B0%84):

![](9_一文详尽之Embedding（向量表示）！_image.we)

关于流式生成模型的原理,感兴趣的自己看:

1.  ICLR 2015 NICE-Non-linear Independent Components Estimation
2.  ICLR 2017 Density estimation using Real NVP
3.  NIPS 2018 Glow: Generative Flow with Invertible 1×1 Convolutions

### **2.3.2 Bert-whitening**

「Bert-whitening」([https://spaces.ac.cn/archives/8069)的作者就是苏神@苏剑林(https://www.zhihu.com/people/su-jian-lin-22),他首先分析了余弦相似度的假定](https://spaces.ac.cn/archives/8069)%E7%9A%84%E4%BD%9C%E8%80%85%E5%B0%B1%E6%98%AF%E8%8B%8F%E7%A5%9E@%E8%8B%8F%E5%89%91%E6%9E%97(https://www.zhihu.com/people/su-jian-lin-22),%E4%BB%96%E9%A6%96%E5%85%88%E5%88%86%E6%9E%90%E4%BA%86%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E5%BA%A6%E7%9A%84%E5%81%87%E5%AE%9A): 余弦相似度的计算为两个向量的内积除以各自的模长,而该式子的局限性为仅在标准正交基下成立.同时,经过假设分析,得出结论: **一个符合“各向同性”的空间就可以认为它源于标准正交基.他觉得Bert-flow虽然有效,但是训练flow模型太麻烦,可以找到一个更简单**的方式实现空间转换.标准正态分布的均值为0、协方差矩阵为单位阵,它的做法就是将Bert向量的均值变换为0、协方差矩阵变换为单位阵,也就是传统数据挖掘中中的白化(whitening)操作.

### **2.3.3 SimCSE**

![](10_一文详尽之Embedding（向量表示）！_image.we)

**对比学习(Contrastive learning)** 的目的是通过将语义上接近的邻居拉到一起,将非邻居推开,从而学习有效的表征(「Hadsell et al., 2006」([https://ieeexplore.ieee.org/document/1640964)).SBert模型本身也是一种对比学习的体现,但是它需要进行有监督训练,依赖标注数据,因此无法在大规模的语料下进行训练.如果能找到一种方法,能够基于文本预料自动生成用于SBert模型训练的数据,就可以解决数据标注依赖的问题.基于对比学习的无监督学习方法,就是通过对文本本身进行传统的文本增广(如转译、删除、插入、调换顺序等等),并将增广后得到的文本作为正样本(即假设文本增广前后,语义相似),同时取同一batch下,其他句子的增广结果作为负样例,就得到了可以进行SBert(或同类模型)训练的预料.而SimCSE(https://arxiv.org/abs/2104.08821)发现,发现利用预训练模型中自带的Dropout](https://ieeexplore.ieee.org/document/1640964)).SBert%E6%A8%A1%E5%9E%8B%E6%9C%AC%E8%BA%AB%E4%B9%9F%E6%98%AF%E4%B8%80%E7%A7%8D**%E5%AF%B9%E6%AF%94%E5%AD%A6%E4%B9%A0**%E7%9A%84%E4%BD%93%E7%8E%B0,%E4%BD%86%E6%98%AF%E5%AE%83%E9%9C%80%E8%A6%81%E8%BF%9B%E8%A1%8C%E6%9C%89%E7%9B%91%E7%9D%A3%E8%AE%AD%E7%BB%83,**%E4%BE%9D%E8%B5%96%E6%A0%87%E6%B3%A8%E6%95%B0%E6%8D%AE**,%E5%9B%A0%E6%AD%A4%E6%97%A0%E6%B3%95%E5%9C%A8%E5%A4%A7%E8%A7%84%E6%A8%A1%E7%9A%84%E8%AF%AD%E6%96%99%E4%B8%8B%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83.%E5%A6%82%E6%9E%9C%E8%83%BD%E6%89%BE%E5%88%B0%E4%B8%80%E7%A7%8D%E6%96%B9%E6%B3%95,%E8%83%BD%E5%A4%9F%E5%9F%BA%E4%BA%8E**%E6%96%87%E6%9C%AC%E9%A2%84%E6%96%99%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E7%94%A8%E4%BA%8ESBert%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E7%9A%84%E6%95%B0%E6%8D%AE**,%E5%B0%B1%E5%8F%AF%E4%BB%A5%E8%A7%A3%E5%86%B3%E6%95%B0%E6%8D%AE%E6%A0%87%E6%B3%A8%E4%BE%9D%E8%B5%96%E7%9A%84%E9%97%AE%E9%A2%98.%E5%9F%BA%E4%BA%8E**%E5%AF%B9%E6%AF%94%E5%AD%A6%E4%B9%A0**%E7%9A%84**%E6%97%A0%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0**%E6%96%B9%E6%B3%95,%E5%B0%B1%E6%98%AF%E9%80%9A%E8%BF%87%E5%AF%B9%E6%96%87%E6%9C%AC%E6%9C%AC%E8%BA%AB%E8%BF%9B%E8%A1%8C%E4%BC%A0%E7%BB%9F%E7%9A%84%E6%96%87%E6%9C%AC%E5%A2%9E%E5%B9%BF(%E5%A6%82%E8%BD%AC%E8%AF%91%E3%80%81%E5%88%A0%E9%99%A4%E3%80%81%E6%8F%92%E5%85%A5%E3%80%81%E8%B0%83%E6%8D%A2%E9%A1%BA%E5%BA%8F%E7%AD%89%E7%AD%89),%E5%B9%B6%E5%B0%86**%E5%A2%9E%E5%B9%BF%E5%90%8E%E5%BE%97%E5%88%B0%E7%9A%84%E6%96%87%E6%9C%AC**%E4%BD%9C%E4%B8%BA**%E6%AD%A3%E6%A0%B7%E6%9C%AC**(%E5%8D%B3%E5%81%87%E8%AE%BE%E6%96%87%E6%9C%AC%E5%A2%9E%E5%B9%BF%E5%89%8D%E5%90%8E,%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC),%E5%90%8C%E6%97%B6%E5%8F%96**%E5%90%8C%E4%B8%80batch%E4%B8%8B,%E5%85%B6%E4%BB%96%E5%8F%A5%E5%AD%90%E7%9A%84%E5%A2%9E%E5%B9%BF%E7%BB%93%E6%9E%9C%E4%BD%9C%E4%B8%BA%E8%B4%9F%E6%A0%B7%E4%BE%8B**,%E5%B0%B1%E5%BE%97%E5%88%B0%E4%BA%86%E5%8F%AF%E4%BB%A5%E8%BF%9B%E8%A1%8CSBert(%E6%88%96%E5%90%8C%E7%B1%BB%E6%A8%A1%E5%9E%8B)%E8%AE%AD%E7%BB%83%E7%9A%84%E9%A2%84%E6%96%99.%E8%80%8C**SimCSE(**https://arxiv.org/abs/2104.08821)%E5%8F%91%E7%8E%B0,%E5%8F%91%E7%8E%B0%E5%88%A9%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E4%B8%AD%E8%87%AA%E5%B8%A6%E7%9A%84**Dropout) mask作为“增广手段”得到的Sentence Embeddings,其质量远好于传统的文本增广方法:

![](11_一文详尽之Embedding（向量表示）！_image.we)

1.  利用模型中的Dropout mask,对每一个样例进行>两次前向传播,得到两个不同的embeddings向量,将同一个样例得到的向量对作为正样本对,对于每一个向量,选取同一个batch中所有样例产生的embeddings向量作为负样本,以此来训练模型.训练损失函数为:

![](12_一文详尽之Embedding（向量表示）！_image.we)

1.  SimCSE也提供了>有监督学习的训练范式,利用NLI标注数据集进行训练,将每一个样例和与其相对的entailment作为正样本对,同一个batch中其他样例的entailment作为负样本,该样本相对应的contradiction作为hard negative进行训练.训练损失函数为:

![](14_一文详尽之Embedding（向量表示）！_image.we)

**3\. 大模型时代的展望**
----------------

随着LLM模型时代的到来,什么才是LLM时代向量模型训练的范式？自动编码or对比学习？我们会持续关注.

![](13_一文详尽之Embedding（向量表示）！_image.we)

参考文献:

1.[https://zhuanlan.zhihu.com/p/374230720](https://zhuanlan.zhihu.com/p/374230720)

2.[https://www.yeahchen.cn/2022/05/14/%25E4%25BB%2580%25E4%25B9%2588%25E6%2598%25AF%25E5%2590%2584%25E5%2590%2591%25E5%25BC%2582%25E6%2580%25A7/](https://www.yeahchen.cn/2022/05/14/%25E4%25BB%2580%25E4%25B9%2588%25E6%2598%25AF%25E5%2590%2584%25E5%2590%2591%25E5%25BC%2582%25E6%2580%25A7/)

3.[https://www.zhihu.com/question/354129879/answer/2471212210](https://www.zhihu.com/question/354129879/answer/2471212210)

4.[https://blog.csdn.net/qq\_48314528/article/details/122760494](https://blog.csdn.net/qq_48314528/article/details/122760494)