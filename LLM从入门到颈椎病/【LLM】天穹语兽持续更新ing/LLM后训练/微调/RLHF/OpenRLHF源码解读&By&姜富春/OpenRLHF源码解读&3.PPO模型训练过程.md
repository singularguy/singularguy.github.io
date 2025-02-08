# OpenRLHF源码解读&3.PPO模型训练过程
> _**作者: 姜富春**_
> 
> _**原文:**_ [_**https://zhuanlan.zhihu.com/p/14569025663**_](https://zhuanlan.zhihu.com/p/14569025663)

0.引语
----

已经用了两篇文章讲解了PPO的源码解读：

1.  [训练整体过程](https://zhuanlan.zhihu.com/p/13043187674)
2.  [经验数据采集过程](https://zhuanlan.zhihu.com/p/14569025663)

最后我们在来看看模型训练过程的一些细节。

1.PPO训练过程
---------

### 1.1. 核心源码

PPO训练过程：[详见PPOtrainer源码的ppo\_train()入口函数](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer.py%23L236C6-L237C47)。核心代码块如下：

```text-plain
class PPOTrainer(ABC):
    ################
    # 1.loss定义 （Actor模型两个loss， Critic模型一个loss）
    ################
    self.actor_loss_fn = PolicyLoss(eps_clip)
    self.critic_loss_fn = ValueLoss(value_clip)
    self.ptx_loss_fn = GPTLMLoss()

    def ppo_train(self, global_steps=0):
        ################
        # 2. 加载经验数据（Experience）
        ################
        dataloader = DataLoader(...)

        for epoch in range(self.max_epochs):
            for experience in pbar:
                ################
                # 3. 执行一步训练
                ################
                status = self.training_step(experience, global_steps)
    
   def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        ################
        # 3.1. 训练Actor 模型，支持2种任务同时训练（SFT和PPO），对应loss GPTLMLoss, PolicyLoss
        ################
        status = self.training_step_actor(experience)
        ################
        # 3.2. 训练Critic 模型，通过Valueloss计算损失
        ################
        status.update(self.training_step_critic(experience))
```

上述代码流程描述可知，PPO训练过程，在一个训练步骤中，Actor和Critic模型依次训练更新。在训练Actor模型时，代码实现中加入了一个可配置的SFT任务，所以Actor是可以同时多任务训练的。具体训练如下图所示。

### 1.2. 模型训练框图

![](OpenRLHF源码解读&3.PPO模型训练过程_image)

> 图1、PPO训练框图

Actor 和Critic 模型结构详见：[姜富春：OpenRLHF源码解读：1.理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674) 第2部分：模型结构部分的网络图。

当前我们基本整理清楚了PPO的完整训练流程。接下来我们进一步看下3个loss函数，理解下模型计算损失的过程。

### 1.3. loss解读

**1.3.1. GPTLMLoss**

GPTLMLoss核心代码块，如下：

```text-plain
# 源码：https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/loss.py#L11C7-L11C16
class GPTLMLoss(nn.Module):
    def __init__(self, ring_attn_group=None):
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

GPTLMLoss就是LLM做next token predict任务的loss（CrossEntropyLoss）。计算loss时，对应 i 位置的预估值logit，取 i+1 位置的token\_id作为label来计算loss。在PPO训练中Actor的SFT任务是个可选的任务。没有这个任务也不影响模型的训练。

**1.3.2. PolicyLoss**

PlicyLoss的核心代码块（看注释）

```text-plain
class PolicyLoss(nn.Module):
    def forward(self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #################
        #1. 重要性采样 important-sampling  
        #   下面公式：(log(p) - log(p')).exp() = log(p/p').exp() = p/p' 
        #   转换下就两个概率的比，表示重要性采样，保证PPO算法是个off-policy算法，提升训练效率 
        #################
        ratio = (log_probs - old_log_probs).exp()
        #################
        # 2. clip-PPO 算法，详见下方公式
        #################
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss
```

这里实现的就是原始论文中的clip-ppo算法，我们把原文公式列在下面：

(1)LCLIP(θ)=Et\[min(rt(θ)At,clip(rt(θ),1−ϵ,1+ϵ)At)\]其中 ：

(2)rt(θ)=πθ(at|st)πθold(at|st)

rt(θ) 是important-sampling的权重，有了这个权重保证了PPO训练可以采样一次训练多次，将[on-policy](https://zhida.zhihu.com/search?content_id=251985503&content_type=Article&match_order=1&q=on-policy&zhida_source=entity)的训练转成off-policy的模式，提升训练效率；At 是经验数据（Experience）中计算好的优势价值打分； ϵ 是clip 超参。代码实现和下面公式完全能对应上， 对Loss的详细理解[参考PPO原论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1707.06347)，不过多赘述。

**1.3.3. ValueLoss**

ValueLoss的核心代码块

```text-plain
class ValueLoss(nn.Module):
    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor) -> torch.Tensor:
        ##############
        # 嚯，计算regrenssion loss(MSE)
        ##############
        loss = (values - returns)  2
```

ValueLoss计算就是对比状态预估价值（values）**和**实际计算的经验价值(returns)的相近程度，典型的回归问题。用MSE（Mean Squared Loss）计算损失。

2.总结
----

本文对PPO采样后的train过程的源码和[Loss函数](https://zhida.zhihu.com/search?content_id=251985503&content_type=Article&match_order=1&q=Loss%E5%87%BD%E6%95%B0&zhida_source=entity)做了详细的讲解。

至此，通过三篇文档已经描述了PPO单机训练的完整过程。其他两篇详见：

1.  [姜富春：OpenRLHF源码解读：1.理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674)
2.  [姜富春：OpenRLHF源码解读：2.PPO训练Experience数据采样过程](https://zhuanlan.zhihu.com/p/14569025663)