# LLM剪枝
*   [万字长文谈深度神经网络剪枝综述](https://zhuanlan.zhihu.com/p/692858636?)

> 目前，大多数针对大模型模型的压缩技术都专注于模型量化领域，即降低单个权重的数值表示的精度。另一种模型压缩方法模型剪枝的研究相对较少，即删除网络元素，包括从单个权重（非结构化剪枝）到更高粒度的组件，如权重矩阵的整行/列（结构化剪枝）。

> 本系列将针对一些常见大模型剪枝方案（LLM-Pruner、SliceGPT、SparseGPT、Wanda等）进行讲述。

*   [大模型剪枝技术原理：概述](https://www.zhihu.com/question/652126515/answer/3457652467)
*   [大模型剪枝技术原理：LLM-Pruner、SliceGPT](https://github.com/liguodongiot/llm-action/blob/main)
*   [大模型剪枝技术原理：SparseGPT、Wanda](https://github.com/liguodongiot/llm-action/blob/main)
*   [大模型剪枝技术原理：总结](https://github.com/liguodongiot/llm-action/blob/main)

**\>结构化剪枝**\>：

*   > LLM-Pruner(LLM-Pruner: On the Structural Pruning of Large Language Models)
    
*   > LLM-Shearing(Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning)
    
*   > SliceGPT: Compress Large Language Models by Deleting Rows and Columns
    
*   > LoSparse
    

**\>非结构化剪枝**\>：

*   > SparseGPT(SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot)
    
*   > LoRAPrune(LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning)
    
*   > Wanda(A Simple and Effective Pruning Approach for Large Language Models)
    
*   > Flash-LLM(Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity)