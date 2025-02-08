# RLHF是什么？
> 作者: Kevin Ren
> 
> 原文: [https://www.zhihu.com/question/644676855/answer/72942689602](https://www.zhihu.com/question/644676855/answer/72942689602)

**提问**：RLHF中PPO有什么问题，为什么大家都设计很多方法去替代它。

RLHF（基于人类反馈的强化学习）中的 PPO（[近端策略优化](https://zhida.zhihu.com/search?content_id=707704596&content_type=Answer&match_order=1&q=%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96&zhida_source=entity)）虽然是目前最常用的算法之一，但它也存在一些固有的问题，这也是为什么研究者们不断探索和设计替代方法的原因。PPO 在 RLHF 中面临的主要挑战有：

**1\. 超参数敏感性 (Hyperparameter Sensitivity)：**

*   **问题：** PPO 的性能_**对超参数的选择非常敏感**_，例如学习率、剪切参数 (clip parameter)、熵系数等。不同的任务和数据集可能需要不同的超参数设置，而_**找到最佳的超参数组合通常需要大量的实验和调参工作**_，这使得 PPO 的应用成本较高。
*   **影响：** 如果超参数设置不当，可能导致训练不稳定、收敛速度慢，甚至无法收敛。
*   **例子：** 剪切参数控制了策略更新的幅度。如果剪切参数过小，会导致策略更新过于保守，收敛速度慢；如果剪切参数过大，会导致策略更新过于激进，容易出现训练不稳定。

**2\. 样本效率相对较低 (Relatively Low Sample Efficiency)：**

*   **问题：** 相比于一些其他的强化学习算法，例如确定性策略梯度算法 (DPG) 或其变体，PPO 的_**样本效率相对较低。这意味着 PPO 需要更多的训练样本才能达到相同的性能水平**_。
*   **原因：** PPO 是一种 _**on-policy**_ 算法，它只能使用_**当前策略生成的数据**_进行训练。这限制了其样本的利用率。
*   **影响：** 在 RLHF 中，获取人类反馈的成本通常很高，因此样本效率是一个重要的考虑因素。如果算法的样本效率较低，就需要更多的人工标注，增加了训练成本。

**3\.** [**奖励函数**](https://zhida.zhihu.com/search?content_id=707704596&content_type=Answer&match_order=1&q=%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0&zhida_source=entity)**设计和分布偏移 (Reward Function Design and Distribution Shift)：**

*   **问题：** RLHF 依赖于奖励模型，而_**奖励模型的设计本身就存在挑战**_。如何设计一个能够准确反映人类偏好的奖励函数是一个难题。此外，_**随着 PPO 训练的进行，生成文本的分布会发生变化**_，这会导致奖励模型的分布偏移，从而影响训练的稳定性。
*   **影响：** 如果奖励函数设计不当或出现分布偏移，PPO 可能会朝着错误的方向优化，导致生成的文本虽然在奖励模型上得分很高，但实际上并不符合人类的期望（即所谓的“奖励欺骗”或“奖励利用”）。

**4\. 训练不稳定 (Training Instability)：**

*   **问题：** 即使使用了剪切技巧，PPO 的训练仍然可能出现不稳定的情况，例如策略震荡、[梯度爆炸](https://zhida.zhihu.com/search?content_id=707704596&content_type=Answer&match_order=1&q=%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8&zhida_source=entity)等。
*   **原因：** PPO 仍然是一种_**基于梯度的优化算法**_，而深度神经网络的训练本身就存在_**不稳定性**_。
*   **影响：** 训练不稳定会导致模型性能波动较大，难以收敛到最优解。

**为什么需要替代方法？**

由于 PPO 存在以上问题，研究者们一直在探索和设计替代方法，以提高 RLHF 的效率、稳定性和[鲁棒性](https://zhida.zhihu.com/search?content_id=707704596&content_type=Answer&match_order=1&q=%E9%B2%81%E6%A3%92%E6%80%A7&zhida_source=entity)。这些替代方法主要集中在以下几个方面：

*   **提高样本效率：** 例如，使用 _**off-policy 算法或**_[_**模型预测控制**_](https://zhida.zhihu.com/search?content_id=707704596&content_type=Answer&match_order=1&q=%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B%E6%8E%A7%E5%88%B6&zhida_source=entity)_**等方法**_。
*   **提高训练稳定性：** 例如，使用更先进的优化算法或正则化技术。
*   **简化训练流程：** 例如，_**直接优化策略**_而无需显式地训练奖励模型。

**一些替代方法**

以下是一些旨在替代或改进 PPO 的方法：

*   **直接偏好优化 (DPO)：** DPO 是一种不需要显式奖励模型的对齐方法，它可以直接根据人类的偏好数据优化策略。DPO 被认为更稳定、更易于训练，并且不需要像 PPO 那样进行大量的超参数调整。
*   **隐式语言模型奖励 (ILM)：** ILM 尝试_**从预训练语言模型本身提取奖励信号**_，从而避免了训练单独的奖励模型。
*   **其他 off-policy 算法：** 例如，SAC (Soft Actor-Critic) 等 off-policy 算法具有更高的样本效率，但也可能引入其他问题，例如方差较大。

PPO 虽然是 RLHF 中一个重要的里程碑，但它并非完美无缺。超参数敏感性、样本效率相对较低、奖励函数设计和分布偏移、训练不稳定等问题促使研究者们不断探索新的方法。DPO 等替代方法的出现为 RLHF 提供新的思路和方向。

【PPO过程四个模型及之间的关系】

**它们之间的关系**

1.  学生（Policy Model）根据题目（输入/Prompt）写作文（生成文本）。
2.  老师（Value Model）根据学生的作文给出评价和指导（价值/Value），预测文章的“未来表现”。
3.  评委（Reward Model）根据学生的作文给出具体的得分（奖励/Reward），代表了文章的“即时表现”，反映了文章是否符合人类的偏好。
4.  学生在修改作文时，会参考老师的评价和评委的得分，并与之前的草稿（Reference Policy Model）进行比较，以确定修改的方向和幅度。
5.  通过不断地练习和修改，学生（Policy Model）的写作水平会不断提高，争取在比赛中获得更高的分数。

**更技术化的解释：**

*   Policy Model 生成文本，Value Model 评估文本的长期价值，Reward Model 评估文本的即时奖励。
*   PPO 算法使用 Reward Model 提供的奖励信号来更新 Policy Model，并使用 Value Model 来估计优势函数，从而更有效地进行策略更新。
*   Reference Policy Model 用于计算 KL 散度，限制 Policy Model 的更新幅度，防止训练不稳定。

* * *

**回答：**

1.  Notable Complexity: 由于PPO中需要4个模型同时加载在GPU中，policy model，ref policy model，value model，reward model。所以会占用很多GPU机器。
2.  Online learning problem, 除此之外，由于模型是online 采样，在policy过batch samples的时候，reward model会空置，在reward model给pair打分的时候，policy model也会空置，那么GPU利用率会不高。
3.  PPO的调超参数会比较困难，需要一些炼丹高手和经验去做。