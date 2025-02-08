# LLM预训练
**MTP:**
--------

通过解码阶段的优化，将1-token的生成，转变成multi-token的生成，从而提升训练和推理的性能。具体来说，在训练阶段，一次生成多个后续token，可以一次学习多个位置的label，进而有效提升样本的利用效率，提升训练速度；在推理阶段通过一次生成多个token，实现成倍的推理加速来提升推理性能。有两篇好文章，如下

[**DeepSeek中的Multi-Token Prediction**](https://zhuanlan.zhihu.com/p/21277164237)

[**deepseek技术解读(2)-MTP（Multi-Token Prediction）的前世今生**](https://zhuanlan.zhihu.com/p/18056041194)

* * *