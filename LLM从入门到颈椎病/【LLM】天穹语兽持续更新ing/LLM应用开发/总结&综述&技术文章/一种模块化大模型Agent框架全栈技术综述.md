# 一种模块化大模型Agent框架全栈技术综述
* * *

created: 2025-01-24T00:20 updated: 2025-01-26T02:20
---------------------------------------------------

> _**作者: PaperAgent**_
> 
> _\*\*原文: \*\*_[_**https://mp.weixin.qq.com/s/raB8HGPOjML3FRZ9NANmNQ**_](https://mp.weixin.qq.com/s/raB8HGPOjML3FRZ9NANmNQ)

现有基于LLM的智能体虽然在功能上取得了进展，但**缺乏模块化**，导致在研究和开发中存在**术语和架构上的混淆**，在软件架构上缺乏统一。

“A survey on LLM based autonomous agents”提出的框架，它并没有明确指出大型语言模型（LLM）、工具、数据源和记忆是否是Agent的一部分。这种对每个模块功能的模糊区分促进了**软件开发者之间的分裂**，并导致**不兼容和阻碍了可重用性**

![](一种模块化大模型Agent框架全栈技术综述_image.pn)

**LLM-Agent-UMF框架**通过明确区分智能体的不同组件，包括LLM、工具和新引入的核心智能体（**core-agent**），来解决这些问题。核心智能体是智能体的中央协调器，包含**规划、记忆、档案、行动和安全**五个模块，其中安全模块在以往的研究中常被忽视。

**核心智能体作为基于大型语言模型（LLM）智能体的中心组成部分**

![](4_一种模块化大模型Agent框架全栈技术综述_image.we)

**核心智能体（core-agent）的内部结构**

核心智能体（core-agent）是LLM-Agent-UMF框架的关键组成部分。核心智能体被设计为智能体的**中央协调器**，负责管理和协调智能体的各种功能和组件。内部结构被划分为五个主要模块，每个模块都有其特定的功能和责任：

**核心智能体的内部结构**

![](一种模块化大模型Agent框架全栈技术综述_image.we)

1.  **规划模块（Planning Module）**：
    *   规划模块是核心智能体的关键组成部分，负责将复杂的任务分解成可执行的步骤，并生成有效的行动计划：
    *   **规划过程（Planning Process）**：
        *   **任务分解（Task Decomposition）**：将复杂任务分解为更简单的子任务，建立中间目标的层次结构。
        *   **计划生成（Plan Generation）**：为每个子任务制定具体计划，包括所需工具和参与方。
    *   **规划策略（Planning Strategies）**：
        *   **单路径策略（Single-path Strategy）**：生成单一路径或程序序列来实现目标，不探索替代方案。
        *   **多路径策略（Multi-path Strategy）**：生成多个计划，评估并选择最合适的路径。
    *   **规划技术（Planning Techniques）**：
        *   **基于规则的技术（Rule-based Technique）**：使用符号规划器和PDDL等正式推理方法。
        *   **语言模型驱动的技术（Language Model Powered Technique）**：利用LLM的知识和推理能力来制定规划策略。
    *   **反馈源（Feedback Sources）**：
        *   **人类反馈（Human Feedback）**：来自核心智能体与人类的直接互动，用于调整规划以符合人类价值观和偏好。
        *   **工具反馈（Tool Feedback）**：来自核心智能体使用的内部或外部工具的反馈，用于优化工具选择和使用策略。
        *   **同级核心智能体反馈（Sibling Core-Agent Feedback）**：来自同一系统内不同核心智能体之间的互动和信息交换。
2.  **记忆模块（Memory Module）**：
    *   负责存储和检索与核心智能体活动相关的信息，以提高决策效率和任务执行能力。
    *   记忆结构分为短期记忆和长期记忆，分别对应不同的信息存储和检索需求。
    *   记忆位置包括嵌入式记忆（核心智能体内）和记忆扩展（核心智能体外，但在智能体系统内）。
    *   记忆格式可以是自然语言、嵌入向量、SQL数据库或结构化列表。
3.  **档案模块（Profile Module）**：
    *   定义LLM的角色和行为，以适应特定的用例和策略。
    *   包含多种方法，如手工制作上下文学习方法、LLM生成方法、数据集对齐方法和新引入的微调可插拔模块方法。
4.  **行动模块（Action Module）**：
    *   将智能体的决策转化为具体行动，通过行动目标、行动产生、行动空间和行动影响四个视角来定义。
    *   行动产生方法包括通过记忆回忆、计划遵循和API调用请求来执行行动。
5.  **安全模块（Security Module）**：
    *   监控行动模块，特别是在生产环境中，以确保LLM的安全和负责任的使用。
    *   遵循机密性、完整性、可用性（CIA）原则，确保信息和资源的安全。
    *   安全措施包括提示保护、响应保护和数据隐私保护。

**核心智能体（core-agent）的分类**

对核心智能体进行了分类，区分为**主动核心智能体（Active Core-Agents）**和**被动核心智能体（Passive Core-Agents）**，以阐明它们在结构和功能上的差异。

**主动和被动核心智能体的内部结构**

![](5_一种模块化大模型Agent框架全栈技术综述_image.we)

![](1_一种模块化大模型Agent框架全栈技术综述_image.we)

**主动核心智能体（Active Core-Agents）：**

*   包含规划、记忆、档案、行动和安全五个模块。
*   负责协调和管理智能体的其他组件，需要规划模块来分解任务、提供上下文、分析信息和做决策。
*   具有状态性（stateful），能够维护关于其过去交互和状态的信息。
*   能够控制LLM的行为和档案，具有动态适应不同任务的能力。
*   在多核心智能体系统中，可能需要复杂的同步机制。

**多主动核心智能体架构**

![](8_一种模块化大模型Agent框架全栈技术综述_image.we)

**被动核心智能体（Passive Core-Agents）：**

*   主要负责执行特定程序，通常不包含规划和记忆模块。
*   通常是无状态的（stateless），只处理当前任务的状态。
*   行动模块是其核心，根据外部指令（如LLM或主动核心智能体的指令）执行操作。
*   与人类的互动通常是单向的，只能由被动核心智能体发起。
*   在多核心智能体系统中，集成新的核心智能体相对简单，因为它们主要执行特定的、有限的任务。

**包括被动核心智能体的基于大型语言模型（LLM）的智能体架构**

![](3_一种模块化大模型Agent框架全栈技术综述_image.we)

**多被动核心智能体架构**

![](6_一种模块化大模型Agent框架全栈技术综述_image.we)

**混合多核心智能体（Hybrid Multi-Core Agent）架构，**

*   这是一种结合了主动核心智能体（Active Core-Agents）和被动核心智能体（Passive Core-Agents）的智能体设计。
*   利用主动核心智能体的管理和协调能力，以及被动核心智能体的执行特定任务的能力。
*   在保持系统灵活性和可扩展性的同时，处理更广泛的任务。

**一主动多被动核心智能体混合架构**

![](2_一种模块化大模型Agent框架全栈技术综述_image.we)

**多主动多被动核心智能体混合架构**

![](11_一种模块化大模型Agent框架全栈技术综述_image.we)

**核心智能体（core-agent）的有效性**

*   验证LLM-Agent-UMF框架在设计和改进多核心智能体系统中的应用价值。
*   展示如何通过合并不同智能体的特性来创建具有增强功能的新型智能体。
*   通过将LLM-Agent-UMF框架应用于现有的智能体，如Toolformer、Confucius、ToolLLM和ChatDB，来识别和分类这些智能体中的核心智能体及其模块。

**使用LLM-Agent-UMF对最新智能体进行分类。**

![](9_一种模块化大模型Agent框架全栈技术综述_image.we)

**Toolformer和Confucius的多被动核心智能体系统**：结合了Toolformer和Confucius的被动核心智能体，以处理特定的工具调用和任务执行。

**基于大型语言模型的智能体1（LA1）：Toolformer和Confucius - 多被动核心智能体架构。**

![](7_一种模块化大模型Agent框架全栈技术综述_image.we)

**ToolLLM和ChatDB的多主动核心智能体系统**：将ToolLLM的API检索能力和ChatDB的复杂推理能力结合起来，创建了一个能够执行高级任务规划和执行的智能体。

**基于大型语言模型的智能体2-A（LA2-A）：ToolLLM和ChatDB - 多主动核心智能体架构。**

![](10_一种模块化大模型Agent框架全栈技术综述_image.we)

[https://arxiv.org/pdf/2409.11393LLM-AGENT-UMF](https://arxiv.org/pdf/2409.11393LLM-AGENT-UMF): LLM-BASED AGENT UNIFIED MODELING FRAMEWORK FOR SEAMLESS INTEGRATION OF MULTI ACTIVE/PASSIVE CORE-AGENTS

**推荐阅读**

*   • [对齐LLM偏好的直接偏好优化方法：DPO、IPO、KTO](http://mp.weixin.qq.com/s?__biz=Mzk0MTYzMzMxMA==&mid=2247484447&idx=1&sn=f01188d29e2c5133addbd67229db4ee7&chksm=c2ce3e6ef5b9b77874aa250e55522bbbaf214df817ad5f5f1ff98135255863522daeebdf2d3b&scene=21#wechat_redirect)
*   • [一篇搭建AI大模型应用平台架构的全面指南](http://mp.weixin.qq.com/s?__biz=Mzk0MTYzMzMxMA==&mid=2247488458&idx=1&sn=672e92203d2ffa05db06967d37f7d492&chksm=c2ce29bbf5b9a0addba2e869ee2622fe28b149f03e1e2abcfb911becb91393fa6c19aa883e85&scene=21#wechat_redirect)
*   • [RAG全景图：从RAG启蒙到高级RAG之36技，再到终章Agentic RAG！](http://mp.weixin.qq.com/s?__biz=Mzk0MTYzMzMxMA==&mid=2247487375&idx=1&sn=e16bc2fdaac04e58e99cfd2a1dc0b0cb&chksm=c2ce35fef5b9bce80dcf0a70b753707036fe7f962d7888d60396490841a538b0f46e952f53f0&scene=21#wechat_redirect)
*   • [Agent到多模态Agent再到多模态Multi-Agents系统的发展与案例讲解（1.2万字，20+文献，27张图）](http://mp.weixin.qq.com/s?__biz=Mzk0MTYzMzMxMA==&mid=2247485322&idx=1&sn=71ffb345fca514aa5ce2848cb2c9f071&chksm=c2ce3dfbf5b9b4edd5b98e45c6179890bdea748fb5220636d25f42006954ea5c81afa8735725&scene=21#wechat_redirect)