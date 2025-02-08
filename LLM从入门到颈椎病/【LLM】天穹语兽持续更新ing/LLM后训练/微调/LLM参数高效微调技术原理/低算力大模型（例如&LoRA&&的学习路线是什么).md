# 低算力大模型（例如&LoRA&&的学习路线是什么)
> _**作者：guolipa**_  
> _**链接：**_[_**https://www.zhihu.com/question/593383416/answer/2982213896**_](https://www.zhihu.com/question/593383416/answer/2982213896)

当前以 ChatGPT 为代表的预训练语言模型（PLM）规模变得越来越大，在消费级硬件上进行全量微调（Full Fine-Tuning）变得不可行。此外，为每个下游任务单独存储和部署微调模型变得非常昂贵，因为微调模型与原始预训练模型的大小相同。**参数高效微调方法（Parameter-Efficient Fine-Tuning，PEFT）** 方法被提出来解决这两个问题，**PEFT 可以使 PLM 高效适应各种下游应用任务，而无需微调预训练模型的所有参数**。微调大规模 PLM 所需的资源成本通常高得令人望而却步。在这方面，**PEFT 方法仅微调少量或额外的模型参数，固定大部分预训练参数，大大降低了计算和存储成本**，同时最先进的 PEFT 技术也能实现了与全量微调相当的性能。

**Huggface 开源的一个高效微调大模型的库**[**PEFT**](https://link.zhihu.com/?target=https%3A//github.com/huggingface/peft)，该算法库支持以下四类方法：

LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.09685) Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://link.zhihu.com/?target=https%3A//aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2110.07602.pdf) P-Tuning: [GPT Understands, Too](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385) Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.08691)

**LLM-Adapters**[\[1\]](#ref_1)\*\* 是对 PEFT 库的扩展\*\*，是一个简单易用的框架，将各种适配器集成到 LLM 中，可针对不同的任务执行 LLM 的基于适配器的 PEFT 方法，除了 PEFT 支持的 LoRA、Prefix Tuning、P-Tuning、Prompt Tuning 方法外，主要扩增了 AdapterH、AdapterP 和 Parallel 三种方法。

AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1902.00751.pdf) AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2005.00052.pdf) Parallel: [Towards a Unified View of Parameter-Efficient Transfer Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2110.04366.pdf)

如下图所示，PEFT 方法可以分为三类，不同的方法对 PLM 的不同部分进行下游任务的适配：

*   **Prefix-Tuning / Prompt-Tuning**：在模型的输入或隐层添加 kk 个额外可训练的前缀 tokens（这些前缀是连续的伪 tokens，不对应真实的 tokens），只训练这些前缀参数；
*   **Adapter-Tuning**：将较小的神经网络层或模块插入预训练模型的每一层，这些新插入的神经模块称为 adapter（适配器），下游任务微调时也只训练这些适配器参数；
*   **LoRA**：通过学习小参数的低秩矩阵来近似模型权重矩阵 WW 的参数更新，训练时只优化低秩矩阵参数。

![](3_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

Transformer 结构和最先进的 PEFT 方法

一. Prefix Tuning[\[2\]](#ref_2)[\[3\]](#ref_3)
----------------------------------------------

**Prefix-Tuning 在模型输入前添加一个连续的且任务特定的向量序列**（continuous task-specific vectors），称之为**前缀（prefix）**。前缀被视为一系列“虚拟 tokens”，但是它由不对应于真实 tokens 的自由参数组成。与更新所有 PLM 参数的全量微调不同，**Prefix-Tuning 固定 PLM 的所有参数，只更新优化特定任务的 prefix**。因此，在生产部署时，只需要存储一个大型 PLM 的副本和一个学习到的特定任务的 prefix，每个下游任务只产生非常小的额外的计算和存储开销。

![](4_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

Fine-tuning 更新所有 PLM 参数，并且需要为每个任务存储完整的模型副本。Prefix-tuning 冻结了 PLM 参数并且只优化了 prefix。因此，只需要为每个任务存储特定 prefix，使 Prefix-tuning 模块化且节省存储空间。

如下图所示，以 GPT2 的自回归语言模型为例，将输入 xx 和输出 yy 拼接为 z=\[x;y\]z=\[x;y\] ，经过 LM 的某一层计算隐层表示 h=\[h1,...,hi,...hn\]h=\[h\_{1},...,h\_{i},...h\_{n}\] ， hi=LMϕ(zi,h<i)h\_{i} = LM\_{\\phi}(z\_{i},h\_{<i}) ，其中， XidxX\_{idx} 和 YidxY\_{idx} 分别为输入和输出序列的索引。

![](5_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

Prefix-Tuning 示例图

Prefix-Tuning 在输入前添加前缀，即 z=\[Prefix,x,y\]z=\[Prefix, x,y\] ， PidxP\_{idx} 为前缀序列的索引， |Pidx||P\_{idx}| 为前缀的长度。前缀索引对应着由 θ\\theta 参数化的向量矩阵 PθP\_{\\theta} ，维度为 |Pidx|×dim(hi)|P\_{idx}| \\times dim(h\_{i}) 。隐层表示的计算如下式所示，若索引为前缀索引 PidxP\_{idx} ，直接从 PθP\_{\\theta} 复制对应的向量作为 hih\_{i} （**在模型每一层都添加前缀向量**）；否则直接通过 LM 计算得到，同时，经过 LM 计算的 hih\_{i} 也依赖于其左侧的前缀参数PθP\_{\\theta}，即**通过前缀来影响后续的序列隐层激化值**。

![](6_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

但是直接优化 PθP\_{\\theta} 会导致训练不稳定，通过一个更小的矩阵 Pθ′P\_{\\theta}^{'} 和一个更大的前馈神经网络 MLPθMLP\_{\\theta} 对 PθP\_{\\theta} 进行重参数化: Pθ\[i,:\]=MLPθ(Pθ′\[i,:\])P\_{\\theta}\[i,:\]=MLP\_{\\theta}(P\_{\\theta}^{'}\[i,:\]) 。在训练时，LM 的参数 ϕ\\phi 被固定，只有前缀参数 θ\\theta 为可训练的参数。训练完成后，只有前缀 PθP\_{\\theta} 被保存。

在实验中，作者进行了两组方法变体的对比分析：

**\[ Full vs Embedding-only \]：Embedding-only 方法只在 embedding 层添加前缀向量并优化，而 Full 代表的 Prefix-tuning 不仅优化 embedding 层添加前缀参数，还在模型所有层的激活添加前缀并优化。实验得到一个不同方法的表达能力增强链条：discrete prompting < embedding-only < prefix-tuning**。同时，Prefix-tuning 可以直接修改模型更深层的表示，避免了跨越网络深度的长计算路径问题。

\*\*\[ Prefix-tuning vs Infix-tuning \]：\*\*通过将可训练的参数放置在 xx 和 yy 的中间来研究可训练参数位置对性能的影响，即 \[x;Infix;y\]\[x;Infix;y\] ，这种方式成为 Infix-tuning。实验表明 **Prefix-tuning 性能好于 Infix-tuning**，因为 prefix 能够同时影响 xx 和 yy 的隐层激活，而 infix 只能够影响 yy 的隐层激活。

![](7_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

同时，研究表明前缀的 embedding 使用词表中真实单词的激活来初始化明显优于随机初始化。

二. P-Tuning[\[4\]](#ref_4)
--------------------------

P-Tuning 的方法思路与 Prefix-Tuning 很相近，P-Tuning 利用少量连续的 embedding 参数作为 prompt **使 GPT 更好的应用于 NLU 任务，而 Prefix-Tuning 是针对 NLG 任务设计**，同时，**P-Tuning 只在 embedding 层增加参数，而 Prefix-Tuning 在每一层都添加可训练参数**。

如下图所示，具体的 NLU 任务以预测一个城市的首都为例，一个离散的 prompt 模板 TT 可以写为："The capital of Britain is \[MASK\]."，其中"Britain"为输入的上下文 xx ，"\[MASK\]"位置为需要输出的目标 yy 。而对于连续的 prompt 模板可以表示为： T={\[P0:i\],x,\[Pi+1:m\],y}T={\[P\_{0:i}\],x,\[P\_{i+1:m}\],y} ，其中， \[Pi\]\[P\_{i}\] 表示模板 TT 中 ithi^{th} 个 prompt token，且为伪 token。经过嵌入层将模板 TT 映射为： {h0,...,hi,e(x),hi+1,...,hm,e(y)}{h\_{0},...,h\_{i},e(x),h\_{i+1},...,h\_{m},e(y)} ，其中 hih\_{i} 为可训练的参数，而其它预训练的真实token向量以及模型权重参数都被固定。

![](8_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

P-Tuning 示例图

直接优化连续的 prompt 参数面临两个挑战：一是预训练模型原始的词向量已经高度离散，若随机初始化 prompt 向量并进行 SGD 优化，也只会在小范围内优化并陷入局部最小值；二是 prompt 向量之间是相互关联而不是独立的。论文中**设计了一个 prompt 编码器，该编码器由一个 Bi-LSTM 和一个两层的前馈神经网络组成，对 prompt embedding 序列进行编码后再传入到语言模型中**。

![](9_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

论文的实验主要表明了：在 SuperGLUE 基准测试中，P-tuning 使得 GPT-style 的生成式模型与相同大小的 BERT 在 NLU 方面实现可比较，有时甚至更好的性能。

**P-Tuning V2**[\[5\]](#ref_5) 方法的思路其实和 Prefix-Tuning 相似，在**模型的每一层都应用连续的 prompts** 并对 prompts 参数进行更新优化。同时，该方法是**针对 NLU 任务优化和适配**的。

![](10_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

P-Tuning V2 示例图

三. Prompt Tuning[\[6\]](#ref_6)
-------------------------------

Prompt Tuning 方式可以看做是 Prefix Tuning 的简化，**固定整个预训练模型参数，只允许将每个下游任务的额外 kk 个可更新的 tokens 前置到输入文本中，也没有使用额外的编码层或任务特定的输出层**。如下图所示，在模型大小增加到一定规模时，仅仅使用 Prompt Tuning 就足以达到 Fine Tuning 的性能。

![](11_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

T5 的 Model Tuning 实现了强大的性能，但需要为每个任务存储单独的模型副本。 随着模型规模的增加，对 T5 的 Prompt Tuning 与 Model Tuning 的性能相当，同时允许所有任务复用同一固定模型。Prompt Tuning 方法明显优于使用 GPT-3 的少样本 Prompt Design。

Prompt Tuning 以 T5 为基础，将所有任务转化成文本生成任务，表示为 Prθ(Y|X)Pr\_{\\theta}(Y|X) 。Prompt Tuning 在输入 XX 前额外添加一系列特殊 tokens PP，输入语言模型生成 YY，即 Prθ;θP(Y|\[P;X\])Pr\_{\\theta;\\theta\_{P}}(Y|\[P;X\])。其中，θ\\theta 为预训练模型参数，在训练过程被固定， θP\\theta\_{P} 为 prompts 的专有参数，在训练过程被更新优化。通过将输入 XX 的 embedding 矩阵 XeX\_{e} 与 prompts 的 embedding 矩阵进行拼接 \[Pe,Xe\]\[P\_{e}, X\_{e}\] 输入 T5 模型，最大化 YY 的概率训练模型，但是只有 prompt 参数被更新。

![](1_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

模型微调需要制作整个预训练模型的任务特定副本，推理分批执行。Prompt tuning 只需为每个任务存储一个 Task Prompts，并使用原始预训练模型进行混合任务推理。

\*\*Prompt Tuning 提出了 Prompt Ensembling 方法来集成预训练语言模型的多种 prompts。\*\*通过在同一任务上训练 NN 个 prompts，为一个任务创建了 NN 个单独的模型，同时在整个过程中共享核心的预训练语言建模参数。 除了大幅降低存储成本外，提示集成还使推理更加高效。 处理一个样例时，可以执行批此大小为NN的单个前向传递，而不是计算 NN 次不同模型的前向传递，跨批次复制样例并改变 prompts。在推理时可以使用 major voting 方法从 prompt ensembling 中得到整体的预测。

四. Adapter Tuning
-----------------

与 Prefix Tuning 和 Prompt Tuning 这类在输入前可训练添加 prompt embedding 参数来以少量参数适配下游任务，**Adapter Tuning 则是在预训练模型内部的网络层之间添加新的网络层或模块来适配下游任务**。假设预训练模型函数表示为 ϕw(x)\\phi _{w}(x) ，对于 Adapter Tuning ，添加适配器之后模型函数更新为： ϕw,w0(x)\\phi_{w,w\_{0}}(x) ， ww 是预训练模型的参数， w0w\_{0} 是新添加的适配器的参数，在训练过程中， ww 被固定，只有 w0w\_{0} 被更新。 |w0|≪|w||w\_{0}|\\ll|w| ，这使得不同下游任务只需要添加少量可训练的参数即可，节省计算和存储开销，同时共享大规模预训练模型。

Adapter 主要包括 Series Adapter（串行） 和 Parallel Adapter（并行）：

*   Series Adapter[\[7\]](#ref_7) 的适配器结构和与 Transformer 的集成如下图（a）所示。适**配器模块被添加到每个 Transformer 层两次：多头注意力映射之后和两层前馈神经网络之后**。适配器是一个 bottleneck（瓶颈）结构的模块，由一个两层的前馈神经网络（由向下投影矩阵、非线性函数和向上投影矩阵构成）和一个输出输出之间的残差连接组成。
*   Parallel Adapter 如下图（b）所示。**将适配器模块与每层 Transformer 的多头注意力和前馈层并行计算集成**。

![](12_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

五. LoRA[\[8\]](#ref_8)
----------------------

现有的 PEFT 方法主要有两类：Adapter Tuning 和 Prefix Tuning。**Adapter Tuning 在 PLM 基础上添加适配器层会引入额外的计算，带来推理延迟问题**；而 **Prefix Tuning 难以优化，其性能随可训练参数规模非单调变化，更根本的是，为前缀保留部分序列长度必然会减少用于处理下游任务的序列长度**。

**1\. 方法原理**

给定一个由 Φ\\Phi 参数化的预训练的自回归语言模型 PΦ(y|x)P\_{\\Phi}(y|x) ，对于全量微调，模型参数由预训练权重 Φ0\\Phi\_{0} 初始化，并反复跟随使得条件语言模型目标函数最大化的梯度更新至 Φ0+ΔΦ\\Phi\_{0} + \\Delta\\Phi 。

maxΦ⁡∑(x,y)∑t=1|y|log(PΦ(yt|x,y<t))\\mathop{max}\\limits\_{\\Phi}\\sum\_{(x,y)}\\sum\_{t=1}^{|y|}log(P\_{\\Phi}(y\_{t}|x,y\_{<t})) \\

全量微调的一个主要缺点就是针对每个下游任务都学习和预训练权重维度相同的全新参数集合 ΔΦ\\Delta\\Phi ，即 |ΔΦ|=|Φ0||\\Delta\\Phi|=|\\Phi\_{0}| 。尤其是 GPT-3 175B 这类大模型，全量微调对计算和存储资源的消耗是非常大的，存储和部署不同微调模型实例也是不可能的。LoRA 论文提出了一种计算和存储高效的低秩（Low-Rank）表示方法，利用更小规模的参数集合 Θ\\Theta 来对任务特定的参数增量进行编码， ΔΦ=ΔΦ（Θ）,|Θ|≪|Φ0|\\Delta\\Phi=\\Delta\\Phi（\\Theta）,|\\Theta|\\ll|\\Phi\_{0}| 。利用该方法对 175B GPT-3 微调，需要训练更新的参数数量 |Θ||\\Theta| 可以小到全量微调参数数量 |Φ0||\\Phi\_{0}| 的 0.01%。

具体地，Transformer 等神经网络包含许多执行矩阵乘法的密集层，这些权重矩阵通常具有满秩。研究[\[9\]](#ref_9)表明预训练的语言模型具有较低的"内在维度（Instrisic Dimension）"，并且可以和完整参数空间一样进行有效学习。受此启发，假设权重的更新在微调适配过程中也具有较低的"内在秩（Instrisic Rank）"。对于预训练模型的权重矩阵 W0∈Rd×kW\_{0} \\in R^{d\\times k} ，通过低秩分解（Low-Rank Decomposition）来表示约束其更新。

W0+ΔW=W0+BAW\_{0} + \\Delta W = W\_{0} + BA \\

其中, B∈Rd×r,A∈Rr×k,r≪min(d,k)B \\in R^{d \\times r}, A \\in R^{r \\times k}, r\\ll min(d,k) 。训练过程， W0W\_{0} 被固定不再进行梯度更新，只训练 AA 和 BB ，如下图所示。对于输入 xx ，模型的前向传播过程 h=W0xh=W\_{0}x 被更新为：

h=W0x+ΔWx=W0x+BAxh=W\_{0}x+\\Delta Wx=W\_{0}x+BAx \\

![](13_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

LoRA重参数化示意图，预训练模型的参数W固定，只训练A和B参数

**2\. 方法优点**

*   **全量微调的一般化**：LoRA 不要求权重矩阵的累积梯度更新在适配过程中具有满秩。当对所有权重矩阵应用 LoRA 并训练所有偏差时，将 LoRA 的秩 rr 设置为预训练权重矩阵的秩，就能大致恢复了全量微调的表现力。也就是说，随着增加可训练参数的数量，训练 LoRA 大致收敛于训练原始模型。
*   **没有额外的推理延时**：在生产部署时，可以明确地计算和存储 W=W0+BAW=W\_{0}+BA，并正常执行推理。当需要切换到另一个下游任务时，可以通过减去 BABA 来恢复 W0W\_{0}，然后增加一个不同的 B′A′B'A' ，这是一个只需要很少内存开销的快速运算。最重要的是，与结构参数上微调的模型相比，LoRA 推理过程中没有引入任何额外的延迟。
*   **减少内存和存储资源消耗**：对于用 Adam 训练的大型 Transformer，若 r≪dmodelr\\ll d\_{model} ，LoRA 减少 2/3 的VRAM 用量（训练模型时，模型参数往往都会存储在显存 VRAM 中），因为不需要存储已固定的预训练参数 W0W\_{0} 的优化器状态，可以用更少的GPU进行大模型训练。在 GPT-3 175B 上，训练期间的 VRAM 消耗从 1.2TB 减少到 350GB。在 r=4r=4 且只有query 和 value 矩阵被调整的情况下，checkpoint 的大小大约减少了 10,000 倍（从 350GB 到 35MB）。另一个好处是，可以在部署时以更低的成本切换任务，只需更换 LoRA 的权重，而不是所有的参数。可以创建许多定制的模型，这些模型可以在将预训练的权重存储在 VRAM 中的机器上进行实时切换。在 GPT-3 175B 上训练时，与完全微调相比，速度提高了25%，因为我们不需要为绝大多数的参数计算梯度。

**3\. 实验分析**

实验将五种方法进行对比，包括：Fine-Tuning (全量微调)、Bias-only or BitFit（只训练偏置向量）、Prefix-embedding tuning (PreEmbed，上文介绍的 Prefix Tuning 方法，只优化 embedding 层的激活)、Prefix-layer tuning (PreLayer，Prefix Tuning 方法，优化模型所有层的激活)\*\*、\*\*Adapter tuning（不同的 Adapter 方法：AdapterH\\rm Adapter^{H}[\[10\]](#ref_10)、AdapterL\\rm Adapter^{L}[\[11\]](#ref_11)、 AdapterP\\rm Adapter^{P}[\[12\]](#ref_12)、 AdapterL\\rm Adapter^{L} 、AdapterD\\rm Adapter^{D}[\[13\]](#ref_13) ）

实验结果以 LoRA 在 GPT-3 175B 上的验证分析为例。如下表所示，**LoRA 在三个数据集上都能匹配或超过微调基准，证明了 LoRA 方法的有效性**。

![](14_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

GPT-3 上不同适配方法性能。展示了 WikiSQL 上的逻辑形式验证精度，MultiNLI-matched 上的精度，以及SAMSum上 的 Rouge-1/2/L 值。LoRA 比之前的方法表现更好，包括全量微调。在 WikiSQL 上的结果有 ±0.5% 左右的波动，MNLI-m 有 ±0.1% 左右的波动，SAMSum 有 ±0.2/±0.2/±0.1 左右的三个指标

但是，**并不是所有方法都能从拥有更多的可训练参数中获益，而 LoRA 表现出更好的可扩展性和任务性能**。当使用超过256个特殊token进行 Prefix-embedding tuning 或使用超过32个特殊 tokens 进行 Prefix-layer tuning时，可以观察到性能明显下降。

![](15_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

GPT-3 175B 准确率与WikiSQL和MNLI匹配的几种适配方法的可训练参数数的关系

**\[ 应该对 Transformer 中的哪些权重矩阵应用 LoRA ?\]** 把所有的参数放在 ΔWq\\Delta W\_{q} 或 ΔWk\\Delta W\_{k} 中会导致性能明显下降，同时适配 WqW\_{q} 和 WvW\_{v} 会产生最好的结果。这表明，即使 r=4r=4 的较小秩也能在 ΔW\\Delta W 中捕捉到足够的信息，因此，**适配更多的权重矩阵比适配具有较大秩的单一类型的权重矩阵更可取**。

![](2_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

在可训练参数量相同的情况下，将LoRA应用于GPT-3中不同类型的注意力权重后，WikiSQL和MultiNLI的验证准确率

**\[ LORA的最佳秩是什么? \] LoRA 在很小的 rr 下已经有了很好的表现了（适配 {Wq,Wv}{W\_{q},W\_{v}} 比只适配 WqW\_{q} 更有竞争力）。这表明更新矩阵 ΔW\\Delta W 可能有一个很小的 "intrinsic rank"，增加秩 rr 不一定能够覆盖一个更有意义的子空间，一个低秩的适配矩阵已经足够**。

![](16_低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

在WikiSQL和MultiNLI上用不同的秩 r 进行验证的准确性

\*\*\[适配矩阵 **ΔW\\Delta W** 与 **WW** 关系如何？\] \*\*通过计算 UTWVTU^{T}WV^{T} 将 WW 投射到 ΔW\\Delta W 的 rr 维子空间， U/VU/V 是 ΔW\\Delta W 的左/右奇异向量矩阵。然后，比较 ||UTWVT||F||U^{T}WV^{T}||_{F} 和 ||W||F||W||_{F} 之间的 Frobenius 范数。作为比较，还计算了将 ||UTWVT||F||U^{T}WV^{T}||\_{F}中 UU 、 VV 替换为 WW 的前 rr 个奇异向量或一个随机矩阵。

*   与随机矩阵相比， ΔW\\Delta W 与 WW 有更强的相关性，表明 ΔW\\Delta W 放大了 WW 中已有的一些特征；
*   ΔW\\Delta W 没有重复 WW 的顶级奇异方向，而只是放大了 WW 中没有强调的方向；
*   **低秩适配矩阵可能会放大特定下游任务的重要特征，而这些特征在一般的预训练模型中没有得到强调**。

![](低算力大模型（例如&LoRA&&的学习路线是什么)_imag.webp)

不同秩下 Frobenius 范式

参考
--

1.  [^](#ref_1_0)LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models [https://arxiv.org/abs/2304.01933](https://arxiv.org/abs/2304.01933)
2.  [^](#ref_2_0)Prefix-Tuning: Optimizing Continuous Prompts for Generation [https://aclanthology.org/2021.acl-long.353.pdf](https://aclanthology.org/2021.acl-long.353.pdf)
3.  [^](#ref_3_0)P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks [https://arxiv.org/pdf/2110.07602.pdf](https://arxiv.org/pdf/2110.07602.pdf)
4.  [^](#ref_4_0)GPT Understands, Too [https://arxiv.org/pdf/2103.10385.pdf](https://arxiv.org/pdf/2103.10385.pdf)
5.  [^](#ref_5_0)P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks [https://arxiv.org/pdf/2110.07602.pdf](https://arxiv.org/pdf/2110.07602.pdf)
6.  [^](#ref_6_0)The Power of Scale for Parameter-Efficient Prompt Tuning [https://arxiv.org/pdf/2104.08691.pdf](https://arxiv.org/pdf/2104.08691.pdf)
7.  [^](#ref_7_0)Parameter-Efficient Transfer Learning for NLP [https://arxiv.org/pdf/1902.00751.pdf](https://arxiv.org/pdf/1902.00751.pdf)
8.  [^](#ref_8_0)LoRA: Low-Rank Adaptation of Large Language Models [https://arxiv.org/pdf/2106.09685.pdf](https://arxiv.org/pdf/2106.09685.pdf)
9.  [^](#ref_9_0)Intrinsic Dimensionality Expythons the Effectiveness of Language Model Fine-Tuning [https://arxiv.org/pdf/2012.13255.pdf](https://arxiv.org/pdf/2012.13255.pdf)
10.  [^](#ref_10_0)Parameter-Efficient Transfer Learning for NLP [https://arxiv.org/pdf/1902.00751.pdf](https://arxiv.org/pdf/1902.00751.pdf)
11.  [^](#ref_11_0)Exploring Versatile Generative Language Model Via Parameter-Efficient Transfer Learning [https://aclanthology.org/2020.findings-emnlp.41.pdf](https://aclanthology.org/2020.findings-emnlp.41.pdf)
12.  [^](#ref_12_0)Adapter- Fusion: Non-destructive task composition for transfer learning [https://arxiv.org/pdf/2005.00247.pdf](https://arxiv.org/pdf/2005.00247.pdf)
13.  [^](#ref_13_0)AdapterDrop: On the Efficiency of Adapters in Transformers [https://arxiv.org/pdf/2010.11918v1.pdf](https://arxiv.org/pdf/2010.11918v1.pdf)