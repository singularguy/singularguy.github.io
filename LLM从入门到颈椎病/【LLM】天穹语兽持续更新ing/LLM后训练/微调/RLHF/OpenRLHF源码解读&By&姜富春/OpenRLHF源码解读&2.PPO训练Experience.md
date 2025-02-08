# OpenRLHF源码解读&2.PPO训练Experience数据采样过程
> _**作者: 姜富春**_
> 
> _**原文:**_ [_**https://zhuanlan.zhihu.com/p/14569025663**_](https://zhuanlan.zhihu.com/p/14569025663)

0\. 引语
------

上一篇文章中『[姜富春：基于OpenRLHF源码理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674)』已经介绍了PPO训练的完整过程，训练过程如图1所示

![](OpenRLHF源码解读&2.PPO训练Experience)

> 图1、PPO训练流程

**PPO训练共分为4阶段**，包括：

1.  准备阶段（准备SFT Model， Reward Model）
2.  模型热启（4个模型，Actor，Critic，Reference， Reward）
3.  在线采集经验数据（Experience）
4.  训练Actor Model 和 Critic model

本文基于[OpenRLHF源码](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/README_zh.md)，对阶段3「在线采集经验数据（Experience）」做详细的讲解，中间配合一些数据图，方便通过[可视化](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E5%8F%AF%E8%A7%86%E5%8C%96&zhida_source=entity)的[数据结构](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84&zhida_source=entity)理解数据采集的过程

1\. 采集经验数据过程简述
--------------

由图1可看到经验数据（Experience）采集过程如下：

![](1_OpenRLHF源码解读&2.PPO训练Experience)

> 图2、PPO采样经验数据过程

**3阶段处理：**

1.  **获取一个Batch 指令数据**：从Dataset中获取一个Batch的Prompt
2.  **生成sequence数据**：拿一个Batch的Prompt数据送入到Actor模型的generate()方法，采样一条或多条结果，组成sequences<Prompt, Answer>数据。
3.  **组装Experience数据**：通过四个模型（Actor， Reference， Critic， Reward）将数据加工成Experience。Experience里面维护了多个Tensor域，为下一步训练Actor，Critic模型做准备。

下面通过配合一些源码和数据图，详细讲解下从拿到一个Batch的Prompt数据到最终获取Experience数据的过程

2\. 经验数据采集步骤详解
--------------

### 2.1. 关键代码块

首先从源码中截取关键的代码块（[ppo\_trainer.py](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer.py)）

```text-plain
class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.
    """
    def __init__(self,...) -> None:
        # 采样Experience的类实例
        self.experience_maker = NaiveExperienceMaker(
            actor,critic,reward_model,initial_model,
            tokenizer,prompt_max_len,
            self.kl_ctl,strategy,
            remote_rm_url,reward_fn,
        )

    def fit(
        self,prompts_dataloader,...) -> None:
        for episode in range(start_episode, args.num_episodes):
            for rand_prompts in self.prompts_dataloader:
                ###################
                # 1. Experience采样过程
                ###################
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(rand_prompts, self.generate_kwargs)
                ):
                    self.replay_buffer.append(experience)

                ###################
                # 2. PPO训练过程
                ###################
                status = self.ppo_train(steps)
                ...
```

从源码看，[NaiveExperienceMaker.make\_experience\_list](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py%23L118)是采样Experience的核心方法，该方法将输入的batch\_prompt经过处理后，组装生成Experience数据。

下面我们看下make\_experience\_list的核心代码。（看代码注释）

```text-plain
def make_experience_list(self, all_prompts: Union[str, List[str]], generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        ####################
        # 1. 调用Actor generate()方法获取Prompt的生成结果，把结果存储到Sample对象
        ####################
        samples_list = self.generate_samples(all_prompts, generate_kwargs)
        torch.distributed.barrier()
        ####################
        # 2. 调用make_experience 对每个Sample做处理，组装Experience部分字段（除了advantage和return）
        ####################
        experiences = []
        for samples in samples_list:
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)
        ####################
        # 3. 通过从后往前回溯计算的方式，获取advantage和return值
        ####################
        for experience, reward in zip(experiences, rewards):
            num_actions = experience.info["num_actions"]
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
        return experiences
```

为了进一步讲清楚数据采样的过程，先针对源码中几个数据结构做下说明。源码中一共有两个主要的数据类。

**2.2. 数据类型描述：Sample 和 Experience**

[描述数据](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E6%8F%8F%E8%BF%B0%E6%95%B0%E6%8D%AE&zhida_source=entity)shape的符号说明：

*   B: [batch\_size](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=2&q=batch_size&zhida_source=entity)
*   S: Sequence\_len，是一个Batch padding后的Prompt + response的长度
*   A: num\_actions, 是生成的token长度
*   [Sample](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py%23L88)

注：Sample有两种数据[存储格式](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F&zhida_source=entity) **batched or** packed，batched 格式是默认的，是做了padding对齐的格式; 而packed格式是非padding对齐的连续存储的格式，本文主要以batched数据格式为例，描述[数据处理](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86&zhida_source=entity)过程。

sample数据类定义如下（数据域含义看注释）

```text-plain
@dataclass
class Samples:
    sequences: torch.Tensor                     # Prompt 和 response，shape[B, S]
    attention_mask: Optional[torch.LongTensor]  # attention mask，标识去掉padding有效的attention位置，shape[B, S] 
    action_mask: Optional[torch.BoolTensor]     # action_mask, 标识有效的生成token（去除生成部分组Batch的padding），shape[B, A]
    num_actions: Union[int, torch.Tensor]       # num_actions, 表示action_mask的长度 int
    response_length: torch.Tensor               # response部分 token的数量，shape[B,]
    total_length: torch.Tensor                  # sequences 所有token（prompt + response）所有token的数量，shape[B,]
```

*   **Experience**

Experience数据类定义如下（数据域含义看注释）

```text-plain
@dataclass 
class Experience:
    sequences: torch.Tensor                     # 同Sample的sequences定义，shape[B, S]
    action_log_probs: torch.Tensor              # action 计算log(softmax(logits))的结果，shape[B, A]
    values: torch.Tensor                        # critic 模型预估的当前状态打分预估值，shape[B, A]
    returns: Optional[torch.Tensor]             # returns 按gae方法计算的平衡偏差和方差的状态打分，shape[B, A]
    advantages: Optional[torch.Tensor]          # 按gae方法计算的优势得分值，shape[B, A]
    attention_mask: Optional[torch.LongTensor]  # attention mask，同Sample定义，shape[B, S]
    action_mask: Optional[torch.BoolTensor]     # action_mask，同Sample定义，shape[B, A]
    info: Optional[dict]                        # 保留一些中间信息，shape[B, A]
    kl: Optional[torch.Tensor] = None           # 计算Actor预估分布和reference预估的分布的KL散度，shape[B, A]
```

我们注意到上面的数据描述中，出现了action 和 action\_num，在[语言模型](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)中，action 怎么理解呢？ 我们用一条sequence数据，描述下在语言模型中： si (状态) , ai (动作)的具体的含义。如图3所示

蓝色块：表示Prompt的token  
红色块：表示生成的有效token  
绿色块：表示eos生成结束token

![](2_OpenRLHF源码解读&2.PPO训练Experience)

> 图3、LLM中状态、动作的描述

我们注意到上图，状态序列和动作序列错开一位，因为先有状态才能采取动作进入下一个状态，所以初始prompt就是第一个初始状态。基于prompt生成的第一个token是第一个动作，然后前序token+当前生成的token作为下一个状态。

语言模型中动作 a 和状态 s 描述为：

*   状态 si ：是从初始token到 i 位置的**token序列**
*   动作 ai : 是基于 si 状态序列，生成的**下一个token**

到此，把一些数据结构和[生成模型](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B&zhida_source=entity)中的状态、动作都描述清楚了，下面我们通过一系列数据图，串起来完整的采样Experience数据的过程。

### 2.3. Batch Prompt数据 -> Sample数据

```text-plain
[[https]]://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py#L181C9-L181C77

samples_list = self.generate_samples(all_prompts, generate_kwargs)
```

上面generate\_samples是把Batch Prompt数据处理成Sample数据的实现。下面基于几步图化操作描述下处理过程

**2.3.1. 基于args.micro\_rollout\_batch\_size的配置，将数据做micro\_batch 处理**

比如当前Batch = 8 ， micro\_rollout\_batch\_size = 4 。

则数据处理如下

![](3_OpenRLHF源码解读&2.PPO训练Experience)

> 图4、batch -> micro\_rollout\_batch

下面为了描述方便，我们只以一个micro\_rollout\_batch=4(上图的micro\_rollout\_batch 1)为例，描述后续数据处理过程

**2.3.2. 调用tokenize\_fn，将Prompt token化，padding做左对齐处**

![](4_OpenRLHF源码解读&2.PPO训练Experience)

> 图5、Tokenizer处理

注： 生成模型的Batch[处理数据](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE&zhida_source=entity)，都采用'left'模式对齐，方便并行化做[decoder](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=decoder&zhida_source=entity)过程

**2.3.3. 调用Actor.generate()方法生成sequences，attention\_mask, action\_mask**

详见[actor.generate()定义](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py%23L122)

sequences，attention\_mask, action\_mask几个数据图示化如下

*   **sequences**

![](5_OpenRLHF源码解读&2.PPO训练Experience)

> 图6、sequences 数据

*   **attention\_mask**

![](6_OpenRLHF源码解读&2.PPO训练Experience)

> 图7、attention mask 数据（非padding置1）

*   **action\_mask**

```text-plain
# action_mask处理过程
state_seq = sequences[:, input_len - 1 : -1]
action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
action_mask[:, 0] = 1
```

![](7_OpenRLHF源码解读&2.PPO训练Experience)

> 图8、action mask 数据（实际是对有效状态位置1）

action\_mask矩阵shape=\[B, A\]，也就是序列长度是生成token数(num\_actions)，**实现中action\_mask实际是对有效状态位置值1** （整体按num\_actions长度，向前平移1位）

**2.3.4. 数据封装成Sample**

上面已经描述清楚Sample的关键域：sequences， attention\_mask，action\_mask，num\_actions。可以按Sample 定义封装到数据类内。

经过上述步骤，已经把一个Batch的Prompt 处理成了Sample数据，接下来看看Sample数据进一步封装成Experience数据的处理。

### 2.4. Sample数据 -> Experience数据

```text-plain
[[https]]://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py#L265

self.make_experience(samples).to_device("cpu")
```

上面make\_experience方法是把Sample数据处理成Experience数据的过程。下面描述代码里的几个关键步骤。

**2.4.1. 通过Actor模型计算action\_log\_probs （Experience.action\_log\_probs）**

Actor[模型结构](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84&zhida_source=entity)，详见：[姜富春：基于OpenRLHF源码理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674) 2.1部分

```text-plain
action_log_probs = self.actor(sequences, num_actions, attention_mask)
```

action\_log\_probs的数据视图如下：

注：灰色虚线块，表示不存在的块，画出完整的sequence是为了方便理解数据的生效位置

![](8_OpenRLHF源码解读&2.PPO训练Experience)

> 图9， action\_log\_probs数据图示

action\_log\_probs 是为了计算KL的中间变量。每个token位置，首先按词表维度（vocab\_size）计算softmax，再取log， 最后根据label token的token\_id取到该位置的log\_probs值。由于probs是 (0,1) 的值，取log，是 (−∞,0) 区间的值。所以上面图中的值都是负数。

**2.4.2. 通过Reference模型计算base\_action\_log\_probs**

Actor模型和Reference模型结构是一样的，数据操作逻辑也是一样的，**同2.4.1. 操作**，不赘述。base\_action\_log\_probs也是为了计算KL散度的中间变量

**2.4.3. 计算action\_log\_probs 和base\_action\_log\_probs 的KL散度(Experience.kl)**

这里计算KL散度，并没有实际用两个[概率分布](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83&zhida_source=entity)(词表长度)，然后通过KL的公式计算（KL计算可参见：[姜富春：如何理解分类任务的loss-交叉熵损失](https://zhuanlan.zhihu.com/p/11762918249)）。而是使用了一种轻量的[近似方法](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E8%BF%91%E4%BC%BC%E6%96%B9%E6%B3%95&zhida_source=entity)计算的KL散度。详见： [Approximating KL Divergence](https://link.zhihu.com/?target=http%3A//joschu.net/blog/kl-approx.html)。

```text-plain
# 源码：https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/utils.py#L7

def compute_approx_kl(log_probs: torch.Tensor, log_probs_base: torch.Tensor,...) -> torch.Tensor:
    log_ratio = log_probs.float() - log_probs_base.float()
    log_ratio = -log_ratio
    log_ratio = log_ratio.exp() - 1 - log_ratio
```

**2.4.4. 通过Critic模型计算状态节点的预估价值 (Experience.value)**

详见[CriticModel实现](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py%23L233C11-L233C22)

Critic是预估状态的价值，看代码实现时，参考图3，先理解LLM中状态的起始位置。最终状态序列长度是num\_actions(生成token的数量)，状态序列起始位置是Prompt的最后一个token，结束位置是最后eos token 前一个token， 所以计算出的Critic预估状态价值的数据为：

![](9_OpenRLHF源码解读&2.PPO训练Experience)

> 图10、Critic模型预估状态价值数据

**2.4.5. 通过Reward模型，计算Batch中每个序列的打分 (Experience.info.r)**

详见[RewardModel实现](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py%23L160C5-L160C10)

在RLHF中，Reward Model是一个ORM（outcome Reward Model） 也就是对完整的生成response输出一个打分。代码实现上取每个sequence eos token位置的预估打分值。如图11，图中"xx"也是会并行计算出的[Reward值](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=Reward%E5%80%BC&zhida_source=entity)，单最终只取了序列最后eos位置的score作为完整序列的打分值。最后reward处理成\[B, 1\]格式，每个序列一个打分。

![](10_OpenRLHF源码解读&2.PPO训练Experience)

> 图11、序列Reward打分数据

调用([cumpute\_reward方法](https://link.zhihu.com/?target=https%3A//github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py%23L197C22-L197C36))将Reward值还原到[二维空间](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E4%BA%8C%E7%BB%B4%E7%A9%BA%E9%97%B4&zhida_source=entity)并赋值到eos位置，其他位置都清零0（为下一步计算优势奖励值做准备）。如图12所示

![](11_OpenRLHF源码解读&2.PPO训练Experience)

> 图12、Reward做scatter操作

**2.4.6. 计算优势奖励值（Experience.advantages）和 状态奖励值（Experience.returns）**

计算优势奖励值（advantage）有多种方法，代码中有\["gae", "reinforce", "rloo"\] 三种实现，本文只沿着"gae"的计算方式做了梳理，其他两种方式有机会再整理下，不影响对整体流程的理解。

gae(Generalized Advantage Estimation)是PPO论文中实现的优势奖励值计算方法，可平衡优势预估的偏差和方差，这里不展开方法细节，详见：[原始PPO论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1707.06347)。代码注释中有一段较清晰的计算公式

```text-plain
详见源码：https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py#L356
def get_advantages_and_returns(values: torch.Tensor, rewards: torch.Tensor,）
    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
        - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
        + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
```

其中：

*   γ ：是时间步[衰减因子](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E8%A1%B0%E5%87%8F%E5%9B%A0%E5%AD%90&zhida_source=entity)，表示离当前状态越近奖励值影响越大，越远越衰减。默认值：1不衰减。
*   λ ：是平衡取[观测值](https://zhida.zhihu.com/search?content_id=251936479&content_type=Article&match_order=1&q=%E8%A7%82%E6%B5%8B%E5%80%BC&zhida_source=entity)的步数的参数。默认值：0.95
    *   当 λ→1 时， adv1=R1+γR2+γ2R3+...−V1 表示更多用观测值计算，偏差小，方差大
    *   当 λ→0 时， adv1=R1+γV2−V1 表示更多用估计值计算， 偏差大，方差小

阅读源码，**计算advantage 和 return是个从后向前回溯计算的过程**，如图13所示，使用value 和 reward数据，从后向前依次计算advantage 和 return。

![](12_OpenRLHF源码解读&2.PPO训练Experience)

> 图13、advantage从后向前回溯计算过程

至此我们已经收集到了Experience数据类的所有信息。我们再标注下每个数据的处理过程。

```text-plain
@dataclass 
class Experience:
    sequences: torch.Tensor                     # 来源于Sample的sequences，详见2.3.3
    action_log_probs: torch.Tensor              # 详见2.4.1
    values: torch.Tensor                        # 详见2.4.4
    returns: Optional[torch.Tensor]             # 详见2.4.6
    advantages: Optional[torch.Tensor]          # 详见2.4.6
    attention_mask: Optional[torch.LongTensor]  # 来源于Sample的attention_mask，详见2.3.3
    action_mask: Optional[torch.BoolTensor]     # 来源于Sample的attention_mask，详见2.3.3
    info: Optional[dict]                        # 保留一些中间信息
    kl: Optional[torch.Tensor] = None           # 详见2.4.3
```

**3.总结**
--------

本文结合OpenRLHF源码，配合数据处理视图，详细介绍了PPO训练过程采集经验数据（Experience）的过程。配合文章：[姜富春：基于OpenRLHF源码理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674) 阅读PPO的完整训练过程。

有了经验数据（Experience），下一步我们就可以开始模型训练了。训练过程细节：详见：[姜富春：OpenRLHF源码解读：3.PPO模型训练过程](https://zhuanlan.zhihu.com/p/14813158239)。