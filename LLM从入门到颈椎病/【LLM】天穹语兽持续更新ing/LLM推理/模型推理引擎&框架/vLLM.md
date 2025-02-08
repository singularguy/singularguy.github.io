# vLLM
[vLLM中文站](https://vllm.hyper.ai/)

vLLM 是一款专为[大语言模型](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)推理加速而设计的框架，实现了 KV 缓存内存几乎零浪费，解决了内存管理瓶颈问题。

更多 vLLM 中文文档及教程可访问 →[https://vllm.hyper.ai/](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/)

vLLM 是一个快速且易于使用的库，专为[大型语言模型](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity) (LLM) 的推理和部署而设计。

vLLM 的核心特性包括：

*   最先进的[服务吞吐量](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E6%9C%8D%E5%8A%A1%E5%90%9E%E5%90%90%E9%87%8F&zhida_source=entity)
*   使用 **PagedAttention** 高效管理注意力键和值的内存
*   连续批处理传入请求
*   使用 CUDA/HIP 图实现快速执行模型
*   量化： [GPTQ](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2210.17323), [AWQ](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2306.00978), INT4, INT8, 和 FP8
*   优化的 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成
*   推测性解码
*   分块预填充

vLLM 的灵活性和易用性体现在以下方面：

*   [无缝集成](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E6%97%A0%E7%BC%9D%E9%9B%86%E6%88%90&zhida_source=entity)流行的 HuggingFace 模型
*   具有高吞吐量服务以及各种解码算法，包括_并行采样_、[_束搜索_](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E6%9D%9F%E6%90%9C%E7%B4%A2&zhida_source=entity)等
*   支持张量并行和流水线并行的分布式推理
*   流式输出
*   提供与 OpenAI 兼容的 API 服务器
*   支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 以及 AWS Neuron
*   前缀缓存支持
*   支持多 LoRA

欲了解更多信息，请参阅以下内容：

*   [vLLM announcing blog post](https://link.zhihu.com/?target=https%3A//vllm.ai/) (PagedAttention 教程)
*   [vLLM paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2309.06180) (SOSP 2023)
*   [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://link.zhihu.com/?target=https%3A//www.anyscale.com/blog/continuous-batching-llm-inference) by Cade Daniel et al.
*   [vLLM 聚会](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/community/vllm-meetups)

**文档**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E6%2596%2587%25E6%25A1%25A3)
------------------------------------------------------------------------------------------------------------

### **入门**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E5%2585%25A5%25E9%2597%25A8)

[安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation)

[使用 ROCm 进行安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation-with-rocm)

[使用 OpenVINO 进行安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation-with-openvino)

[使用 CPU 进行安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation-with-cpu)

[使用 Neuron 进行安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation-with-neuron)

[使用 TPU 进行安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation-with-tpu)

[使用 XPU 进行安装](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/installation-with-xpu)

[快速入门](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/quickstart)

[调试提示](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/debugging-tips)

[示例](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/getting-started/examples/)

### **部署**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E9%2583%25A8%25E7%25BD%25B2)

[OpenAI 兼容服务器](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/openai-compatible-server)

[使用 Docker 部署](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/deploying-with-docker)

[分布式推理和服务](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/distributed-inference-and-serving)

[生产指标](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/production-metrics)

[环境变量](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/environment-variables)

[使用统计数据收集](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/usage-stats-collection)

[整合](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/integrations/)

[使用 CoreWeave 的 Tensorizer 加载模型](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/tensorizer)

[兼容性矩阵](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/compatibility%2520matrix)

[常见问题解答](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/serving/frequently-asked-questions)

### **模型**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E6%25A8%25A1%25E5%259E%258B)

[支持的模型](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/supported-models)

[添加新模型](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/adding-a-new-model)

[启用多模态输入](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/enabling-multimodal-inputs)

[引擎参数](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/engine-arguments)

[使用 LoRA 适配器](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/using-lora-adapters)

[使用 VLMs](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/using-vlms)

[在 vLLM 中使用推测性解码](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/speculative-decoding-in-vllm)

[性能和调优](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/models/performance-and-tuning)

### **量化**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E9%2587%258F%25E5%258C%2596)

[量化内核支持的硬件](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/supported_hardware)

[AutoAWQ](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/autoawq)

[BitsAndBytes](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/bitsandbytes)

[GGUF](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/gguf)

[INT8 W8A8](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/int8-w8a8)

[FP8 W8A8](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/fp8-w8a8)

[FP8 E5M2 KV 缓存](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/fp8-e5m2-kv-cache)

[FP8 E4M3 KV 缓存](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/quantization/fp8-e4m3-kv-cache)

### **自动前缀缓存**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E8%2587%25AA%25E5%258A%25A8%25E5%2589%258D%25E7%25BC%2580%25E7%25BC%2593%25E5%25AD%2598)

[简介](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/automatic-prefix-caching/introduction-apc)

[实现](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/automatic-prefix-caching/implementation)

[广义缓存策略](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/automatic-prefix-caching/implementation)

### **性能**[**基准测试**](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95&zhida_source=entity)[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E6%2580%25A7%25E8%2583%25BD%25E5%259F%25BA%25E5%2587%2586%25E6%25B5%258B%25E8%25AF%2595)

[vLLM 的基准套件](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/performance-benchmarks/benchmark-suites-of-vllm)

### [**开发者**](https://zhida.zhihu.com/search?content_id=252269912&content_type=Article&match_order=1&q=%E5%BC%80%E5%8F%91%E8%80%85&zhida_source=entity)**文档**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E5%25BC%2580%25E5%258F%2591%25E8%2580%2585%25E6%2596%2587%25E6%25A1%25A3)

[采样参数](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/sampling-parameters)

[离线推理](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/offline-inference/)

*   [LLM 类](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/offline-inference/llm-class)
*   [LLM 输入](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/offline-inference/llm-inputs)

[vLLM 引擎](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-engine/)

[LLM 引擎](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-engine/)

*   [LLMEngine](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-engine/llmengine)
*   [AsyncLLMEngine](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-engine/asyncllmengine)

[vLLM 分页注意力](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention)

*   [输入处理](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23%25E8%25BE%2593%25E5%2585%25A5)
*   [概念](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23%25E6%25A6%2582%25E5%25BF%25B5)
*   [查询](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23%25E8%25AF%25A2%25E9%2597%25AE-query)
*   [键](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23%25E9%2594%25AE-key)
*   [QK](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23qk)
*   [Softmax](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23softmax)
*   [值](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23%25E5%2580%25BC)
*   [LV](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23lv)
*   [输出](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention%23%25E8%25BE%2593%25E5%2587%25BA)

[输入处理](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/input-processing/model_inputs_index)

*   [指南](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/input-processing/model_inputs_index%23%25E6%258C%2587%25E5%258D%2597)
*   [模块内容](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/input-processing/model_inputs_index%23%25E6%25A8%25A1%25E5%259D%2597%25E5%2586%2585%25E5%25AE%25B9)

[多模态](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/multi-modality/)

*   [指南](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/multi-modality/%23%25E6%258C%2587%25E5%258D%2597)
*   [模块内容](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/multi-modality/%23%25E6%25A8%25A1%25E5%259D%2597%25E5%2586%2585%25E5%25AE%25B9)

[Docker 文件](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/dockerfile)

[vLLM 性能分析](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/profiling-vllm)

*   [示例命令和用法](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/profiling-vllm%23%25E5%2591%25BD%25E4%25BB%25A4%25E5%2592%258C%25E4%25BD%25BF%25E7%2594%25A8%25E7%25A4%25BA%25E4%25BE%258B)
*   [离线推理](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/profiling-vllm%23%25E7%25A6%25BB%25E7%25BA%25BF%25E6%258E%25A8%25E7%2590%2586)
*   [OpenAI 服务器](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/developer-documentation/profiling-vllm%23openai-%25E6%259C%258D%25E5%258A%25A1%25E5%2599%25A8)

**社区**[**​**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/%23%25E7%25A4%25BE%25E5%258C%25BA)
------------------------------------------------------------------------------------------------------------

[vLLM 聚会](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/community/vllm-meetups)

[赞助商](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/community/sponsors)

[**索引和表格**](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/indices-and-tables/index)
-------------------------------------------------------------------------------------------------

*   [索引](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/indices-and-tables/index)
*   [模块索引](https://link.zhihu.com/?target=https%3A//vllm.hyper.ai/docs/indices-and-tables/python-module-index)