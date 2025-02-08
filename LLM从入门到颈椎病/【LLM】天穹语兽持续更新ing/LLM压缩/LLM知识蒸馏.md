# LLM知识蒸馏
*   [大模型知识蒸馏概述](https://www.zhihu.com/question/625415893/answer/3243565375)

**\>Standard KD**\>:

> 使学生模型学习教师模型(LLM)所拥有的常见知识，如输出分布和特征信息，这种方法类似于传统的KD。

*   > MINILLM
    
*   > GKD
    

**\>EA-based KD**\>:

> 不仅仅是将LLM的常见知识转移到学生模型中，还涵盖了蒸馏它们独特的涌现能力。具体来说，EA-based KD又分为了上下文学习（ICL）、思维链（CoT）和指令跟随（IF）。

> In-Context Learning：

*   > In-Context Learning distillation
    

> Chain-of-Thought：

*   > MT-COT
    
*   > Fine-tune-CoT
    
*   > DISCO
    
*   > SCOTT
    
*   > SOCRATIC CoT
    

> Instruction Following：

*   > Lion