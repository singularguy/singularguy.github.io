# RLHF： 从Actor-Critic到PPO
> _**作者: Sonsii**_ 
> 
> _**原文:**_ [_**https://zhuanlan.zhihu.com/p/14656217921**_](https://zhuanlan.zhihu.com/p/14656217921)

Actor-Critic
------------

在 Policy Gradient 中，由于使用的是采样得到的累积奖励 $R\_t$，其方差可能较高，导致梯度估计的不稳定。同时，需要大量样本来获得准确的梯度估计，样本效率也很低。

Actor-Critic 是结合了 Policy Gradient（Actor）和价值函数估计（Critic）的方法，同时引入时序差分方法。

*   Actor 是指[策略函数](https://zhida.zhihu.com/search?content_id=251953959&content_type=Article&match_order=1&q=%E7%AD%96%E7%95%A5%E5%87%BD%E6%95%B0&zhida_source=entity)，即学习一个策略以得到尽可能高的回报
*   Critic 是指价值函数 $V\_{\\pi}(s)$，对当前策略的值函数进行估计，即评估 Actor 的好坏

### 推导

在策略梯度更新公式中

$$ \\nabla\\overline{R}_{\\theta} \\approx \\frac{1}{N} \\sum_{n=1}^N \\sum\_{t=1}^{T\_n} \\left(\\sum\_{t'=t}^{T\_n} \\gamma^{t'-t} r\_{t'}^n - b \\right) \\cdot \\nabla \\log p\_\\theta(a\_t^n | s\_t^n) $$

$\\sum\_{t'=t}^{T\_n} \\gamma^{t'-t} r\_{t'}^n$ 这一项代表从 $t'$ 时刻到 $T\_n$ 时刻的累计 Reward，而这恰好就是 $Q$ 函数。因此策略梯度更新可以表示为

$$ \\frac{\\partial \\bar{R}_\\theta}{\\partial \\theta} \\approx \\frac{1}{N} \\sum_{n=1}^{N} \\sum\_{t=1}^{T\_n} \\left( Q\_\\pi (s\_t^n, a\_t^n) - V\_\\theta (s\_t^n) \\right) \\nabla \\log p\_\\theta (a\_t^n | s\_t^n) $$

其中: $$ Q\_\\pi (s\_t^n, a\_t^n) = \\mathbb{E}\[r\_t^n + V\_\\pi (s\_{t+1}^n)\] $$ $$ V\_\\pi (s\_t^n) = \\mathbb{E}_{a \\sim \\pi}\[r\_t^n + \\gamma V_\\pi (s\_{t+1}^n)\] $$

需要同时估计 $Q$ 和 $V$。通过 $Q$ 和 $V$ 的关系做进一步转化: $$ V\_\\pi (s\_t^n) = r\_t^n + V\_\\pi (s\_{t+1}^n) $$ $$ Q\_\\pi (s\_t^n, a\_t^n) = \\mathbb{E}\[r\_t^n + V\_\\pi (s\_{t+1}^n)\] $$

去掉期望

> 经验近似 (一阶时序差分 TD (0))，原始的异步 Actor-Critic 的 paper 尝试了很多方法，最后发现这样最有效

$$ Q^{\\pi}(s\_{t}^{n}, a\_{t}^{n}) = r\_{t}^{n} + V^{\\pi}(s\_{t+1}^{n}) $$

则此时更新公式变成

$$ \\frac{\\partial \\bar{R}_{\\theta}}{\\partial \\theta} \\approx \\frac{1}{N} \\sum_{n=1}^{N} \\sum\_{t=1}^{T\_{n}} \\left( r\_{t}^{n} + V^{\\pi}(s\_{t+1}^{n}) - V\_{\\theta}^{\\pi}(s\_{t}^{n}) \\right) \\nabla \\log p\_{\\theta}(a\_{t}^{n} | s\_{t}^{n}) $$

这样就绕过了对 $Q$ 的估计，虽然引入了新的变量 $r$，但 $r$ 是单步奖励，对比累计奖励 $Q$，$r$ 的方差要小很多。

### 实现技巧

*   [参数共享](https://zhida.zhihu.com/search?content_id=251953959&content_type=Article&match_order=1&q=%E5%8F%82%E6%95%B0%E5%85%B1%E4%BA%AB&zhida_source=entity):

我们需要估计 Actor 网络和 Critic 网络 ($V$ 函数)，二者的输入都是 state，Actor 输出 action 的分布，Critic 输出标量 (价值)，因此二者网络的前面几层可以共享参数

*   探索机制:

探索的方法是对 $\\pi$ 输出的分布设置一个约束，用于使分布的熵不要太小，也就是希望不同的动作被采用的概率平均一些。这样智能体才会多尝试各种不同的动作，把环境探索得比较好。

A3C 算法
------

Actor-Critic 方法存在的主要问题：

1.  **训练不稳定**：单一线程（或单一环境）的训练可能导致梯度更新的相关性较高，进而引起震荡或收敛缓慢。
2.  **低样本效率**：依赖于单个 Agent 的经验，限制了数据采集的速度和多样性。
3.  **探索不足**：单一 Agent 容易陷入局部最优解，探索策略可能受限。
4.  **同步更新瓶颈**：在分布式环境中，单线程的同步更新可能成为训练的瓶颈，限制了并行计算的优势。

A 3 C 针对基础的 Actor-Critic 方法做如下改进：

1.  异步并行训练

A 3 C 通过同时运行多个独立的 Agent，每个 Agent 在不同的环境实例中独立与环境交互。这些 Agent 异步地将梯度更新发送到全局共享的网络参数。多个 Agent 同时采集样本，加快了数据的积累速度。不同 Agent 的独立探索有助于覆盖更广阔的状态空间，避免陷入局部最优。

1.  多步时序差分

A 3 C 使用多步 TD（时序差分）方法，而不是单步更新。这意味着每个 Worker 在更新梯度时，会基于 n 步后的奖励和状态价值进行计算

$$ A\_t = \\sum\_{k=0}^{n-1} \\gamma^k r\_{t+k} + \\gamma^n V\_w (s\_{t+n}) - V\_w (s\_t) $$

多步返回在减少方差的同时，保持了一定的偏差，可以更稳定地估计优势函数。同时，通过利用未来多个时间步的信息，加快了策略的更新步伐。

1.  探索策略的改进

A 3 C 中的各个 Worker 由于在不同时间和环境实例中独立运行，不同 Worker 的独立探索行为增加了策略的多样性，避免了单一 Agent 可能出现的探索不足问题。同时，通过熵正则化，鼓励策略在选择动作时保持一定的随机性，平衡探索与利用。

$$ L = L\_{\\text{policy}} + c \\cdot L\_{\\text{value}} + \\beta \\cdot S(\\pi(\\cdot | s)) $$

其中，$S(\\pi(\\cdot | s))$ 是策略的熵，$\\beta$ 是权重系数。

PPO
---

### 同策略 (on-policy) vs 异策略 (off-policy)

学习的模型和与环境交互的模型是否为同一个

*   同策略

如策略梯度中，Actor 模型与环境交互搜集轨迹 $\\tau$，根据搜集到的轨迹 $\\tau$ 按照策略梯度公式更新 Policy 模型的参数，然后 Actor 重新采样，进行下一轮迭代。因此这里的 Actor 和 Policy 是同一个模型

*   同策略的问题在于采样数据的利用率低。在做完一轮参数更新后，Policy 和 Actor 都有了变化，因此上一轮采样的数据不再能用，需要重新采样。同策略算法会花费大量时间进行采样操作
*   异策略

重要性采样: 假设 $x$ 由分布 $p$ 采样而来，则 $f(x)$ 的期望，可由下式计算：  
$$ E\_{x \\sim p}\[f(x)\] \\approx \\frac{1}{N} \\sum\_{i=1}^{N} f(x\_i) $$

当无法从分布 $p(x)$ 直接采样数据，只能从另一个分布 $q(x)$ 进行采样时，此时，可由下式计算 $f(x)$ 的期望：  
$$ \\int f(x) p(x) dx = \\int f(x) \\frac{p(x)}{q(x)} q(x) dx = E\_{x \\sim q} \\left\[ f(x) \\frac{p(x)}{q(x)} \\right\] $$

则： $$ E\_{x \\sim p}\[f(x)\] = E\_{x \\sim q} \\left\[ f(x) \\frac{p(x)}{q(x)} \\right\] $$

其中 $\\frac{p(x)}{q(x)}$ 即为**重要性权重**，$q(x)$ 可以是任何分布，唯一的限制是当 $p(x)$ 为 0 时 $q(x)$ 不为 0，否则会没有定义。

*   应用到异策略： 采用单独的 Actor 模型 ($\\theta'$) 与环境交互，计算梯度时加入重要性权重：  
    $$ \\nabla \\overline{R}_\\theta = E_{\\tau \\sim p\_{\\theta'}}\[\\frac{p\_{\\theta}(\\tau)}{p\_{\\theta'}(\\tau)} R(\\tau) \\nabla \\log p\_{\\theta}(\\tau)\] $$

### PPO 算法

通过引入重要性采样，我们可以将策略梯度算法由同策略变为异策略。但是重要性采样存在一个问题就是，两个分布的差异不能过大。这就是 PPO 要解决的问题。PPO 通过在优化目标中引入 KL 散度来约束两个分布：

$$ J\_{\\text{PPO}}(\\theta') = J\_{\\theta'}(\\theta) - \\beta KL(\\theta, \\theta') $$

### 近端策略优化惩罚（PPO-penalty）

整体思想同上，不过引入了自适应 KL 惩罚的概念，即设置 $KL\_{\\text{max}}$ 和 $KL\_{\\text{min}}$，当一轮更新结束后，如果 $KL(\\theta, \\theta\_k) > KL\_{\\text{max}}$，增大 $\\beta$，如果 $KL(\\theta, \\theta\_k) < KL\_{\\text{min}}$，减小 $\\beta$：

$$ J\_{\\text{PPO}}(\\theta\_k) = J\_{\\theta\_k}(\\theta) - \\beta KL(\\theta, \\theta\_k) $$

### 近端策略优化裁剪（PPO-clip）

同样的初衷，引入 Clip 操作：

$$ J\_{\\text{PPO}}^{2}(\\theta\_k) = \\sum\_{(s\_t, a\_t)} \\min \\left( \\frac{p\_{\\theta}(a\_t | s\_t)}{p\_{\\theta\_k}(a\_t | s\_t)} A\_{\\theta\_k}(s\_t, a\_t), \\text{clip}\\left( \\frac{p\_{\\theta}(a\_t | s\_t)}{p\_{\\theta\_k}(a\_t | s\_t)}, 1-\\epsilon, 1+\\epsilon \\right) A\_{\\theta\_k}(s\_t, a\_t) \\right) $$

RLHF 中的 PPO
-----------

首先明确 LLM 对齐过程中的强化学习四要素：

*   Policy: 即目标 LLM，用于根据给定 prompt 生成 response
*   State: LLM 生成每个 token 时依赖的 context，即 prompt + 截止 t-1 时刻新生成的 token 序列
*   Action: 根据 context 生成的 token 即为 action，action 空间就对应着词表
*   Reward: 度量生成的 response 对人类偏好的符合程度； $V\_t$ 则代表实际期望总收益（即时+未来），目标 LLM 当下产生 token，到整个 response 生成结束的期望总收益

这里的 PPO 共涉及 4 个模型:

*   Actor: 演员模型，即我们最终希望得到的目标 LLM
*   Critic: 评论家模型，用于预估总收益
*   Reward: 奖励模型，用于计算即时收益
*   Reference: 参考模型，用于在对齐阶段给 LLM 增加一些约束，防止语言模型偏离起点太多，不受控

其中，Actor 和 Critic 是需要训练的，而 Reward 和 Reference Model 的参数是冻结的

### Actor

即我们最终的目标 LLM，一般用 SFT 阶段的模型初始化。整个训练的终极目标是，Actor 可以根据给定的 prompt 生成符合人类偏好的 response，因此，我们的方式是:

1.  先喂给 Actor 一条 prompt，让它生成对应的 response。
2.  对 prompt + response 算得最后的 loss (计算方式见下文)，用于更新 actor。

### Critic

用于预测期望总收益，它需要做参数更新。在不同的框架实现中，Critic 的设计和初始化方式也有多种，例如：

*   和 Actor 共享部分参数
*   从 RW 阶段的 Reward Model 初始化而来（如 deepspeed-chat 的实现）

Critic 的作用可以简单理解为对人类偏好的量化判断。

### Reward

用于计算生成每个 token 的即时收益。经过 RW 阶段的训练，Reward model 已经具备了一定的收益估计能力，在对齐过程中，其估计值被视为准确客观的。 Reward model 很大的作用是辅助 Critic 的预估。对于当前时刻 t 的价值函数 $V\_t$ 的估计，有两种方式:

*   方式一: 完全由 Critic 估计得到
*   方式二: 由 Critic 估计下一时刻价值，结合当前时刻的 Reward 估计值，也即 $V\_t = R\_t + \\gamma V\_{t+1}$

显然，方式二的估计更为准确，因为其包含 Reward model 估计值，而 Reward model 在 RW 阶段已经经过了一次训练，其估计值相对更加准确客观。而方式一完全由未训练好的 Critic 估计，偏差显然更大。

### Reference

通常由 SFT 模型初始化，在训练过程中，参数冻结，其作用是通过 KL 散度为 Actor 增加约束：

1.  将 prompt 喂入 Actor，生成相应的 response，response 中每一个 token 的结果记为 log\_probs
2.  之后，将 prompt 拼接 response 喂入 Reference model，生成每个 token 的 log\_prob 结果，记为 ref\_log\_probs
3.  计算 log\_probs - ref\_log\_probs，即为 KL 散度

### LOSS 计算

涉及两部分 loss：

*   Actor loss：用于评估 Actor 生成的 response 是否符合人类偏好
*   Critic loss：用于评估 Critic 是否正确预测了人类的偏好

### Actor loss

结合 PPO-penalty 和 PPO Clip，引入 GAE（广义优势估计）：

$$ \\text{actor\_loss} = -\\min \\left( A\_t \\cdot GAE \\cdot \\frac{P(A\_t | S\_t)}{P\_{\\text{old}}(A\_t | S\_t)}, A\_t \\cdot GAE \\cdot \\text{clip}\\left( \\frac{P(A\_t | S\_t)}{P\_{\\text{old}}(A\_t | S\_t)}, 1-\\epsilon, 1+\\epsilon \\right) \\right) $$

其中，GAE（广义优势估计）定义如下：

$$ A\_t^{GAE} = \\sum\_{k=0}^{\\infty} (\\gamma \\lambda)^k \\delta\_{t+k} $$

### Critic loss

critic 本质就是对 $V(s\_t)$ 进行预估，即状态 $s\_t$ 下未来能拿到的总收益 return 的期望。因此，我们应该让 $V(s\_t)$ 拟合 return，并计算损失函数来指导参数更新。

Returns 的计算如下：

$$ \\text{returns} = A\_t^{GAE} + V(s\_t) $$

由于：

$$ A\_t^{GAE} = \\sum\_{k=0}^{\\infty} (\\gamma \\lambda)^k \\delta\_{t+k} $$

为方便理解，令 $\\lambda=0$，此时就有：

$$ A\_t^{GAE} = \\delta\_t = r\_t + \\gamma V(s\_{t+1}) - V(s\_t) $$

则：

$$ \\text{return} = r\_t + \\gamma V(s\_{t+1}) $$

至此，RLHF 中 PPO 的理论运作模式介绍完毕。 最后，贴一个结合代码的tutorial，常看常新: [Reinforcement Learning From Human Feedback](https://link.zhihu.com/?target=https%3A//newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)