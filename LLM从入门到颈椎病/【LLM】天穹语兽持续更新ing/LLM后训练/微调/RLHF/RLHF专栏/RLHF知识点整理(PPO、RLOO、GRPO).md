# RLHF知识点整理(PPO、RLOO、GRPO)
> 作者: 
> 
> 原文:

导读 价值 (Value Function) 价值函数（Value Function）用于评估在给定策略下，某个状态或状态-动作对的潜在价值。价值函数可以帮助代理决定哪些状态或动作更有利。常见的价值函数有以下两种：

状态价值函数 (State Value Function)

在策略下，从状态 s 开始的预期累积奖励。(注意: 预期累积奖励指的是未来时刻)

其中是折扣因子，用来减少未来奖励的权重，是时间 t+1 获得的即时奖励。

状态动作价值函数 (Action Value Function) 在策略下，从状态 s 采取动作 a 后的预期累积奖励。

策略 策略是一个从状态到动作的概率分布，决定了代理在给定状态 s 下选择动作 a 的概率。策略可以是确定性的（Deterministic）或随机性的（Stochastic）

确定性策略

在状态 s 下，选择动作 a 的规则。

随机性策略

在状态 s 下，选择动作 a 的概率。

贝尔曼方程 用于描述状态价值函数和状态-动作价值函数之间的递归关系。

状态价值函数的贝尔曼方程：

状态-动作价值函数的贝尔曼方程:

最优价值和最优策略 最优状态价值函数

在所有可能的策略中，从状态 s 开始的最高预期累积奖励。

最优状态-动作价值函数

在所有可能的策略中，从状态 s 采取动作 a 后的最高预期累积奖励。

最优策略

在所有可能的策略中，使得从任意状态开始的预期累积奖励最大化的策略。

价值迭代和策略迭代 价值迭代 (Value Iteration)

价值迭代是一种动态规划方法，通过迭代更新状态价值函数 V (s) 来找到最优价值函数，进而导出最优策略

步骤:

1.  初始化：初始化状态价值函数为任意值（通常是 0）。
2.  迭代更新：对于每个状态，使用贝尔曼最优方程更新价值函数：

其中，是从状态 s 采取动作 a 后转移到状态 s'并获得即时奖励 r 的概率。是折扣因子，。数度 k 次迭代时的状态价值函数。

1.  收敛判断：重复上述步骤，直到价值函数的变化小于一个很小的阈值，即：
2.  导出最优策略:

策略迭代 (Policy Iteration)

策略迭代也是一种动态规划方法，通过交替执行策略评估（Policy Evaluation）和策略改进（Policy Improvement）来找到最优策略 $\\pi^\*$

步骤：

1.  初始化：选择一个任意的初始策略。
2.  策略评估：对于给定的策略，评估其状态价值函数。
3.  策略改进: 根据当前策略的价值函数，通过贪心算法来改进策略
4.  策略收敛: 如果新的策略与旧的策略相同，即，则算法终止，此时的策略即为最优策略。否则，将作为新的策略，返回策略评估步骤。

基于价值的方法（Value-Based Methods）： 定义：基于价值的方法通过学习一个价值函数来指导行动选择。价值函数评估在给定状态下采取特定行动的好坏程度，或者在给定状态下的预期回报。最常用的价值函数是动作价值函数（Action-Value Function），也称为 Q 函数。

工作原理：在基于价值的方法中，算法试图找到一个最优的动作价值函数，该函数可以预测在特定状态下采取特定动作后可以获得的预期回报。一旦找到这个函数，就可以通过选择具有最高预期回报的动作来决定最佳行动。

示例：Q 学习（Q-Learning）和深度 Q 网络（Deep Q-Network, DQN）是最著名的基于价值的方法。

基于策略的方法（Policy-Based Methods）： 定义：基于策略的方法直接学习一个策略函数，该函数映射观察（或状态）到行动，而不需要通过一个中间的价值函数。策略可以是确定性的（对于给定的状态总是选择相同的动作）或随机的（为每个动作分配一个概率）。

工作原理：在这种方法中，目标是直接优化策略函数，以最大化长期奖励。这意味着算法直接调整采取行动的概率，以便在长期中获得更高的奖励。

示例：策略梯度方法（Policy Gradient Methods），如 REINFORCE，以及 Actor-Critic 方法，后者结合了基于价值和基于策略的方法的优点。

演员评论家 （Actor-Critic）：

1.  Actor（行动者）

定义：Actor 是一个策略函数，它根据当前的状态选择动作。这个策略可以是确定性的（对于给定的状态总是选择相同的动作）或随机的（为每个动作分配一个概率）。

目标：Actor 的目标是学习一个最优的策略，使得在给定状态下选择的动作能够最大化长期奖励。

更新方式：Actor 通常通过策略梯度方法进行更新。具体来说，它根据 Critic 提供的反馈来调整其参数，以提高预期奖励。

1.  Critic（评论者）

定义：Critic 是一个价值函数，它评估当前策略的好坏。Critic 通常学习一个状态价值函数（V 函数）或动作价值函数（Q 函数），用于估计在给定状态下采取特定动作的预期回报。

目标：Critic 的目标是准确地估计当前策略的性能，即在给定状态下采取特定动作后的预期回报。

更新方式：Critic 通常通过时序差分（Temporal Difference, TD）学习或蒙特卡洛（Monte Carlo, MC）方法进行更新。这些方法通过比较预测值和实际奖励来调整价值函数的参数。

1.  工作流程

初始状态：环境提供一个初始状态 s 。

选择动作：Actor 根据当前策略选择一个动作 a。

执行动作：环境根据选择的动作 a 进行状态转移，返回新的状态 s' 和奖励 r 。

评估性能：Critic 评估当前状态 s 的价值，提供一个价值估计 V (s) 或 Q (s, a) 。

更新 Critic：Critic 通过比较预测值和实际奖励来更新价值函数。例如，使用 TD 学习：

其中，是学习率，是折扣因子。

更新 Actor：Actor 根据 Critic 提供的反馈来调整策略。例如，使用策略梯度方法： 其中，($\\theta$) 是策略参数，是策略函数，是 Critic 提供的动作价值。

REINFORCE 算法 REINFORCE 是一种基于策略的强化学习算法。与基于价值的方法（如 Q-Learning）不同，REINFORCE 直接学习一个策略函数，该函数定义了在给定状态下采取每个动作的概率。其目标是通过与环境的交互，调整策略参数，以最大化长期奖励的期望值。REINFORCE 算法的核心思想是根据所采取的动作获得的奖励来调整策略。如果一个动作导致了较高的奖励，那么采取该动作的概率应该增加；反之，如果一个动作导致了较低的奖励，那么采取该动作的概率应该减少。这种调整是通过计算策略梯度来实现的，策略梯度指示了如何调整策略参数以提高预期奖励。

算法步骤

初始化：初始化策略参数。

生成轨迹：使用当前策略生成一个或多个从环境开始到结束的完整轨迹（即一系列状态-动作-奖励的序列）。

计算回报：对于每个轨迹，计算从每个状态开始到结束的所有奖励的总和，这称为回报。

计算梯度：根据策略梯度公式计算策略梯度。对于给定的策略，策略梯度可以表示为： 其 中 ， 表示一个完整的轨迹，T 是轨迹的长度。

更新参数：使用梯度上升法更新策略参数，即，其中是学习率。

重复：重复上述步骤，直到策略收敛或达到预设的停止条件。

Ppo 算法 延迟奖赏

当前状态价值包括即时奖励和延迟奖励，不仅需要考虑下一个状态的价值，还需要考虑未来状态的价值。

价值函数 (Value Function)

状态价值

是在状态 s 下遵循当前策略所能获得的期望回报，可以看作是在状态 s 下采取所有可能动作的加权平均值，权重为采取每个动作的概率。

直接计算过于困难 (状态空间太大), 因此通常会使用一个神经网络来进行估计，也就是经常提到的 value model。

动作价值

是在状态下采用行为可以获得所有奖励的期望值。

优势函数 (Adavantage Function)

优势函数用于评估采样某一个动作的好坏。优势函数 $A(s,a)$ 衡量的是在给定状态 s 下采取特点动作 a 的汇报相比于遵循当前策略时平均回报的优势。优势函数可以定义为动作价值函数 (Q 函数) 与状态价值函数 (V 函数) 之间的差值:

其中是在状态 s 下采取动作 a 后，遵循当前策略所能获得的期望回报。

优势函数的作用

基于策略梯度的算法，其目标函数为: ，可以看到只要 Q 值大于 0，那么模型就会向这个方向优化更新。比如在语言模型当中，下一个词的候选项可以是 (你、我、他) 3 个动作，对应的奖励分别是 "10,20,30"，很显然，奖励为 10 的少于平均值 20，应该受到惩罚，而奖励 30 的应该受到鼓励。因此就有了:

其中为优势值，

广义优势估计（Generalized Advantage Estimation, GAE）

在 PPO 中，优势函数通常使用广义优势估计（Generalized Advantage Estimation, GAE）来计算，这是一种结合了蒙特卡洛方法和时间差分学习的技术，旨在减少估计的方差，同时保持估计的无偏性。GAE 的基本思想是将优势函数的估计分解为多个部分，每个部分考虑从当前时刻开始，在未来一定时间范围内的累积回报与值函数的差值。

其中:

1.  是折扣因子，用于减少未来奖励的影响。
2.  是 GAE 的参数，用于平衡蒙特卡洛方法和 TD 学习的影响。
3.  是在时间步 t 获得的即时奖励, 由 reward model 计算得到。
4.  是状态 s 的价值函数
5.  是时间差分残差。

重要性采样

重要性采样的主要目的是在策略更新时，使用旧策略生成的数据来估计新策略的性能，从而避免了每次都从新策略中重新采样数据的需要。重要性采样的基本思想是通过权重调整来补偿不同策略之间的差异。具体来说，假设我们有一个旧策略和一个新策略重要性采样的权重可以表示为： ，其中是旧策略在状态 s 下选择动作 a 的概率，是新策略在状态 s 下选择动作 a 的概率。

Ppo 算法的步骤

1.  收集数据：从当前策略中采样一批轨迹（trajectories），包括状态 s、动作 a、奖励 r 和下一个状态 s'。
2.  计算回报：根据采样的轨迹，计算每个状态-动作对的回报。回报通常定义为从当前时间步 t 开始的未来奖励的折扣和：
3.  计算优势函数：使用价值函数 (V (s)) 来估计每个状态的优势函数 (A (s, a)):

其中是时间差分残差。

1.  计算重要性采样权重：对于每个状态-动作对 ((s, a))，计算重要性采样权重 ：
2.  更新策略：使用重要性采样权重来调整优势函数，并更新策略。PPO 通过以下目标函数来更新策略：

有些地方会写成 (比如: trl)

其中: 是一个剪切操作，确保在和之间，以防止策略更新时出现大的跳跃，从而保持学习的稳定性。

重要性采样权重再范围内 (情况 1 和情况 2) 新旧策略差异不大，此时当优势函数大于 0 时，鼓励增大在状态下采用的概率，当优势函数小于 0 时，减少在状态下采用的概率。 重要性采样权重低于下界。(情况 3 和情况 4) 新旧策略差异较大 (新策略采用的可能性低于旧策略)。如果此时优势函数大于 0，鼓励增大在状态下采用的概率。如果此时优势函数小于 0，此时不应该再增大在状态下采用的概率 (注意: 此时，因此 )。 重要性采样权重高于上界。(情况 3 和情况 4) 新旧策略差异较大 (新策略采用的可能性高于旧策略)。如果此时优势函数大于 0，因为旧的策略已经有较大的概率选择，不希望过于贪心，因此不需要继续鼓励在状态下采用的概率。如果此时优势函数小于 0，应该减少在状态下采用的概率。

RLHF (Reinforcement Learning from Human Feedback,) 按照 trl==0.12.1 实现的 ppo\_trainer 梳理 rlhf 的流程。一套完整的 rlhf 算法一共由 4 个模型组成，分别是:

策略模型 policy model 一个 CausalLM 模型，其输出的 next token 概率分布，就对应强化学习当中的 action。由一堆 transformer-decoder 和一个 lm\_head (是 Linear (hidden\_size, vocab\_size)) 层组成。

奖赏模型 reward model

一个 SequenceClassification 模型，其输出是一个完整的 promp+response 的奖赏值，由一堆 transformer-decoder 和一个 score (是 Linear (hidden\_size, 1)) 层组成。注意: 再标准的 rlhf 当中，奖赏值只会对 response 输出的最后一个 token 计算奖赏值，其它时刻都是 0，因此前面计算优势函数的公式里面的 r 只会对最后一个 token 有值，其它时刻都是 0。

价值模型 value model 一个 SequenceClassification 模型，其输出是 generate 阶段每一个输出 token 的价值估计，由一堆 transformer-decoder 和一个 score (是 Linear (hidden\_size, 1)) 层组成。注意: 再标准的 rlhf 当中，价值模型会对每一个输出的 token 都估计一个值，用于计算优势函数，即:

参考模型 reference model 和 policy model 一样, 一个 CausalLM 模型，会将 reference model 和 policy model 输出的 logits 计算 kl 损失，保证 policy model 和 reference model 差异不至于过大。

流程 准备样本

1.  Policy model 使用输入的 prompt 生成 response（默认温度设置为 0.7）以及对应的 logits，记作。
2.  Value model 使用 prompt 以及 policy model 生成的 response 计算每一个输出 token 的 value, 记作。
3.  Reward model 使用 prompt 以及 policy model 生成的 response 当前 response 的 reward (最后一个 token)，记作。
4.  Reference model 使用 prompt 以及 policy model 生成的 response 计算 response 的 logits，记作。

奖赏值计算

注意：

1.  仅仅是在当前 response 输出的最后一个 token 才会有值，其它情况下都为 0。
2.  用于防止 policy model 与 reference model 输出的概率分布过大。
3.  Trl 的实现当中还会对没有结束符号的 response 相应减少得分。

奖赏值白化

计算优势值

使用 GAE 算法估计优势函数，这一步需要计算每一个输出 token 的估计，因此是比较耗时的一个环节。

优势值白化

重要性采样

策略目标函数

价值目标函数

Trl 最新的实现当中对 value 也进行了裁剪。

最终的目标函数

Rloo\_trainer 参考文献: Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs

Ppo\_trainer 里面实现的 ppo 算法需要 4 个模型，并且其奖赏值是相对于 token 级别的，再使用 GAE 算法估计优势函数的时候必须要遍历整个所有输出的 token，因此从显存的占用、训练时长都不友好。Rloo 算法仅需要: policy model、reward model、reference model 三个模型就能够，可以有效的减少显存占用。

Rloo 使用的是 REINFORCE 优化算法，避开了价值函数直接学习一个策略函数，并且和 ppo 当中计算优势函数的思路类似，它也需要计算奖励的“基线”，直接把 (“奖励”-“基线”) 与生成 token 的对数概率值相乘作为目标函数。再 rloo 计算基线的过程当中，会将本批次内所有其它奖励的均值当做基线。另外 rloo 的奖励是相对于整个 response，不再是 token 级别。因为 rloo 不再需要 value model，因此损失函数就只包含策略目标函数。

计算优势值

其中是本匹次内生成的所有 response。

策略目标函数

GRPO (Group Relative Policy Optimization) 参考文献: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

参考代码:[https://github.com/saisurbehera/trl/blob/grpo/trl/trainer/grpo\_trainer.py#L380](https://github.com/saisurbehera/trl/blob/grpo/trl/trainer/grpo_trainer.py#L380)

Ppo 算法当中的价值函数通常与策略函数的参数量相当，这带来了显存的内存和计算负担。再 ppo 训练过程当中，价值函数通常会被用于计算优势函数的基线，以减少方差。但是再 llm 模型当中，通常需最后一个有效 token 会被分配一个奖赏值 (reward model 仅仅计算最后一个 token 的奖赏值)，这样会使得训练在每一个 token 上都准确的价值函数变得复杂。类似于 RLOO，GRPO 也不需要价值函数，而是通过对同一个问题生成多个输出的平均奖励作为基线。

注意: 上面的 kl 约束项不再放在优势函数当中，而是直接作为损失函数的一部分。

Kl 约束项采用下面的无偏估计版本:

计算优势 GRPO 会对同一个 prompt 生成多个 response，这些 response 称为一个组 group。每一个组内所有的 response 经过奖励函数输出，通过均值-方差归一化计算出来最终的优势函数:

。(可以看到，对于 grpo 同一个 response 内所有的 token 优势值是一样的)

评估 Trl 当中实现的评估方法:

当前策略与参考策略的平均差异
--------------

Metrics\["objective/kl"\] = self.Accelerator.Gather (mean\_kl). Mean (). Item ()

策略的平均熵, 表明策略所选择的行动的随机性。
-----------------------

Metrics\["objective/entropy"\] = self.Accelerator.Gather (mean\_entropy). Mean (). Item ()

Kl 奖赏值
------

Metrics\["objective/non\_score\_reward"\] = self.Accelerator.Gather (mean\_non\_score\_reward). Mean (). Item ()

奖赏值 (包含 kl 奖赏值)
---------------

Metrics\["objective/rlhf\_reward"\] = self.Accelerator.Gather (rlhf\_reward). Mean (). Item ()

奖赏值 (不包含 kl 奖赏值)
----------------

Metrics\["objective/scores"\] = self.Accelerator.Gather (scores.Mean ()). Mean (). Item ()

平滑之后的 kl 差异情况 (说明历史上的学习情况)
--------------------------

Metrics\["policy/approxkl\_avg"\] = self.Accelerator.Gather (approxkl\_stats). Mean (). Item ()

被剪裁的策略更新的平均比例，表示限制策略更新以防止发生较大变化的频率
----------------------------------

Metrics\["policy/clipfrac\_avg"\] = self.Accelerator.Gather (pg\_clipfrac\_stats). Mean (). Item ()

平均策略损失，表明策略的执行情况
----------------

Metrics\["loss/policy\_avg"\] = self.Accelerator.Gather (pg\_loss\_stats). Mean (). Item ()

价值函数的平均值损失，表示预测值与实际奖励之间的差异。
---------------------------

Metrics\["loss/value\_avg"\] = self.Accelerator.Gather (vf\_loss\_stats). Mean (). Item ()

被剪裁的价值函数更新的平均比例，表示限制价值函数更新以防止发生较大变化的频率
--------------------------------------

Metrics\["val/clipfrac\_avg"\] = self.Accelerator.Gather (vf\_clipfrac\_stats). Mean (). Item ()

训练期间策略的平均熵，表明策略行为的多样性。
----------------------

Metrics\["policy/entropy\_avg"\] = self.Accelerator.Gather (entropy\_stats). Mean (). Item ()

当前策略概率与旧策略概率的平均比率，策略变化程度的衡量标准。
------------------------------

Metrics\["val/ratio"\] = self.Accelerator.Gather (ratio\_stats). Mean (). Item ()#

当前策略概率与旧策略概率的方差，策略变化程度的衡量标准。
----------------------------

Metrics\["val/ratio\_var"\] = self.Accelerator.Gather (ratio\_stats). Var (). Item ()

生成的序列结束 (EOS) token 的数量，可以指示完整响应的数量。
------------------------------------

Metrics\["val/num\_eos\_tokens"\] = (responses == processing\_class. Eos\_token\_id). Sum (). Item () 请你对这份文档内容做一些优化 1. 将代码都用$包裹起来让它能识别 latex 公式 2. 其他都不要动内容不要修改 3. 将内容放到 markdown 代码块内