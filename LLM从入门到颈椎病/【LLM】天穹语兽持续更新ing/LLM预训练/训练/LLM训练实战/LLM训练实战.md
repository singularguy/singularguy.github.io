# LLM训练实战
| LLM | 预训练/SFT/RLHF... | 参数  | 教程  | 代码  |
| --- | --- | --- | --- | --- |
| Alpaca | full fine-turning | 7B  | [从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/llm-train/alpaca) |
| Alpaca(LLaMA) | LoRA | 7B~65B | 1.[足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼](https://zhuanlan.zhihu.com/p/619426866)  <br>2\. [使用 LoRA 技术对 LLaMA 65B 大模型进行微调及推理](https://zhuanlan.zhihu.com/p/632492604) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/llm-train/alpaca-lora) |
| BELLE(LLaMA/Bloom) | full fine-turning | 7B  | 1.[基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化](https://zhuanlan.zhihu.com/p/618876472)  <br>2\. [BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试](https://zhuanlan.zhihu.com/p/621128368) | N/A |
| ChatGLM | LoRA | 6B  | [从0到1基于ChatGLM-6B使用LoRA进行参数高效微调](https://zhuanlan.zhihu.com/p/621793987) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/chatglm-lora) |
| ChatGLM | full fine-turning/P-Tuning v2 | 6B  | [使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调](https://zhuanlan.zhihu.com/p/622351059) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/chatglm) |
| Vicuna(LLaMA) | full fine-turning | 7B  | [大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼](https://zhuanlan.zhihu.com/p/624012908) | N/A |
| OPT | RLHF | 0.1B~66B | 1.[一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇](https://zhuanlan.zhihu.com/p/626159553)  <br>2\. [一键式 RLHF 训练 DeepSpeed Chat（二）：实践篇](https://zhuanlan.zhihu.com/p/626214655) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/deepspeedchat) |
| MiniGPT-4(LLaMA) | full fine-turning | 7B  | [大杀器，多模态大模型MiniGPT-4入坑指南](https://zhuanlan.zhihu.com/p/627671257) | N/A |
| Chinese-LLaMA-Alpaca(LLaMA) | LoRA（预训练+微调） | 7B  | [中文LLaMA&Alpaca大语言模型词表扩充+预训练+指令精调](https://zhuanlan.zhihu.com/p/631360711) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/chinese-llama-alpaca) |
| LLaMA | QLoRA | 7B/65B | [高效微调技术QLoRA实战，基于LLaMA-65B微调仅需48G显存，真香](https://zhuanlan.zhihu.com/p/636644164) | [配套代码](https://github.com/liguodongiot/llm-action/tree/main/train/qlora) |
| LLaMA | GaLore | 60M/7B | [突破内存瓶颈，使用 GaLore 一张4090消费级显卡也能预训练LLaMA-7B](https://zhuanlan.zhihu.com/p/686686751) | [配套代码](https://github.com/liguodongiot/llm-action/blob/main/train/galore/torchrun_main.py) |