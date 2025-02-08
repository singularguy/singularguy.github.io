# LLM测评
*   [C-Eval](https://github.com/liguodongiot/ceval)：全面的中文基础模型评估套件，涵盖了52个不同学科的13948个多项选择题，分为四个难度级别。
*   [CMMLU](https://github.com/liguodongiot/CMMLU)：一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。CMMLU涵盖了从基础学科到高级专业水平的67个主题。它包括：需要计算和推理的自然科学，需要知识的人文科学和社会科学,以及需要生活常识的中国驾驶规则等。此外，CMMLU中的许多任务具有中国特定的答案，可能在其他地区或语言中并不普遍适用。因此是一个完全中国化的中文测试基准。
*   [IFEval: Instruction Following Eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)/[Paper](https://arxiv.org/abs/2311.07911)：专注评估大模型遵循指令的能力,包含关键词检测、标点控制、输出格式要求等25种任务。
*   [SuperCLUE](https://github.com/CLUEbenchmark/SuperCLUE)：一个综合性大模型评测基准，本次评测主要聚焦于大模型的四个能力象限，包括语言理解与生成、专业技能与知识、Agent智能体和安全性，进而细化为12项基础能力。
*   [AGIEval](https://github.com/ruixiangcui/AGIEval/)：用于评估基础模型在与人类认知和解决问题相关的任务中的能力。该基准源自 20 项面向普通考生的官方、公开、高标准的入学和资格考试，例如：普通大学入学考试（例如：中国高考（Gaokao）和美国 SAT）、法学院入学考试、数学竞赛、律师资格考试、国家公务员考试。
*   [OpenCompass](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md)：司南 2.0 大模型评测体系。支持的数据集如下：

| 语言  | 知识  | 推理  | 考试  |
| --- | --- | --- | --- |
| 字词释义 | 知识问答 | 文本蕴含 | 初中/高中/大学/职业考试 |
| \- WiC | \- BoolQ | \- CMNLI | \- C-Eval |
| \- SumEdits | \- CommonsenseQA | \- CNLI | \- AGIEval |
| 成语习语 | \- NaturalQuestions | \- AX-b | \- MMILU |
| \- CHID | \- TriviaQA | \- CB-g | \- GAOKAO-Bench |
| 语义相似度 |     | \- RTE | \- CMMLU |
| \- AFQMC |     | \- ANLI | \- ARC |
| \- BUSTM |     | \- Xiezhi |     |
| 指代消解 |     | 常识推理 | 医学考试 |
| \- CLUEWSC |     | \- StoryCloze | \- CMB |
| \- WSC |     | \- COPA |     |
| \- Winograd |     | \- ReCoRD |     |
| 翻译  |     | \- Hellaswag |     |
| \- Flores |     | \- PIQA |     |
| \- IWSLT2017 |     | \- SIQA |     |
| 多语种问答 |     | 数学推理 |     |
| \- TyDi-QA |     | \- MATH |     |
| \- XCOPA |     | \- GSM8K |     |
| 多语种总结 |     | 定理应用 |     |
| \- XLSum |     | \- TheoremQA |     |
|     |     | \- StrategyQA |     |
|     |     | \- SciBench |     |
|     |     | 综合推理 |     |
|     |     | \- BBH |     |

| 理解  | 长文本 | 安全  | 代码  |
| --- | --- | --- | --- |
| 阅读理解 | 长文本理解 | \- CivilComments | \- HumanEval |
| \- CMRC | \- LEval | \- CrowdsPairs | \- HumanEvalX |
| \- CMRC | \- LongBenchmark | \- CValues | \- MBPP |
| \- DRCD | \- GovReports | \- JigsawMultilingual | \- APPs |
| \- MultiRC | \- NarrativeQA | \- TruthfulQA | \- DS1000 |
| \- RACE | \- Qasper |     |     |
| \- DROP |     | 健壮性 |     |
| \- OpenBookQA |     | \- AdvGLUE |     |
| \- SQuAD2.0 |     |     |     |
| 内容总结 |     |     |     |
| \- CSL |     |     |     |
| \- LCSTS |     |     |     |
| \- XSum |     |     |     |
| \- SumScreen |     |     |     |
| 内容分析 |     |     |     |
| EPSTMT |     |     |     |
| LAMBADA |     |     |     |
| TNEWS |     |     |     |

* * *

LongBenchmark: 一个双盲（中英文）多任务基准数据集，旨在评估大语言模型的长上下文处理能力。它包含21个任务，涵盖单文档问答、多文档问答、摘要、小样本学习、合成任务和代码补全等。数据集平均任务长度范围为5k到115k，共包含47500个测试数据。LongBenchmark采用全自动评估方法，旨在以最低的成本衡量和评估模型长上下文的性能。

EvalScope: 魔搭社区官方推出的模型评测与性能基准测试框架，专为多样化的模型评估需求而设计。它支持广泛的模型类型，包括但不限于大语言模型、多模态模型、Embedding模型、Reranker模型和CLIP模型。EvalScope还适用于多种评测场景，如端到端的AGI评测、竞技场模式和模型推理性能压测等，具有多个常用测试基准和评估指标。

LVEval: 一个具备5个长度等级（16k, 32k, 64k, 128k和256k）、最大文本测试长度达到256k的长文本评测集。