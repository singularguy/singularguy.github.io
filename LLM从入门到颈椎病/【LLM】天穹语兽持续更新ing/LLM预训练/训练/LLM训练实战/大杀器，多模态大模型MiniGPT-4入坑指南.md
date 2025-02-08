# 大杀器，多模态大模型MiniGPT-4入坑指南
> 原文: [https://mp.weixin.qq.com/s/Q9vL9C2wgJOwMuRFIXl9QA](https://mp.weixin.qq.com/s/Q9vL9C2wgJOwMuRFIXl9QA)

ChatGPT的发布给大家带来了不少的震撼，而随后发布的GPT-4更是展现了非凡的多模态能力。但是，ChatGPT和GPT4官方公布的细节很少，OpenAI俨然走上了闭源之路，让广大AI从业者又爱又恨。

最近，来自沙特阿拉伯阿卜杜拉国王科技大学的研究团队开源了GPT-4的平民版 MiniGPT-4。他们认为，GPT-4 具有先进的多模态生成能力的主要原因在于利用了更先进的大型语言模型（LLM）。为了研究这一现象，他们提出了 MiniGPT-4。

MiniGPT-4 简介
------------

MiniGPT-4 仅使用一个投影层将一个冻结的视觉编码器（BLIP-2）与一个冻结的 LLM（Vicuna）对齐。

![](3_大杀器，多模态大模型MiniGPT-4入坑指南_image.)

image.png

MiniGPT-4 产生了许多类似于 GPT-4 中新兴的视觉语言能力。比如：根据给定的图像创作故事和诗歌，为图像中显示的问题提供解决方案，教用户如何根据食物照片烹饪，给个手绘草图直接写出网站的代码等。

除此之外，此方法计算效率很高，因为它仅使用大约 500 万个对齐的图像-文本对和额外的 3,500 个经过精心策划的高质量图像-文本对来训练一个投影层。

**BLIP-2 简介**

BLIP-2是一种通用且高效的视觉-语言预训练方法，它可以从现成的冻结预训练图像编码器和冻结大型语言模型中引导视觉-语言预训练。BLIP-2通过一个轻量级的Querying Transformer来弥合模态差距，并在两个阶段进行预训练。第一个阶段从冻结图像编码器引导视觉-语言表示学习。第二个阶段从冻结语言模型中引导视觉-语言生成学习。尽管比现有方法具有显著较少的可训练参数，但BLIP-2在各种视觉-语言任务上实现了最先进的性能。在零样本 VQAv2 上，BLIP-2 相较于 80 亿参数的 Flamingo 模型，使用的可训练参数数量少了 54 倍，性能却提升了 8.7 %。

MiniGPT-4 模型训练原理
----------------

MiniGPT-4 的模型架构遵循 BLIP-2，因此，训练 MiniGPT-4 分两个阶段。

第一个传统预训练阶段使用 4 张 A100 卡在 10 小时内使用大约 500 万个对齐的图像-文本对进行训练。 在第一阶段之后，Vicuna 虽然能够理解图像。 但是Vicuna的生成能力受到了很大的影响。

为了解决这个问题并提高可用性，MiniGPT-4 提出了一种通过模型本身和 ChatGPT 一起创建高质量图像文本对的新方法。 基于此，MiniGPT-4 随后创建了一个小规模（总共 3500 对）但高质量的数据集。

第二个微调阶段在对话模板中对该数据集进行训练，以显著提高其生成的可靠性和整体的可用性。 令人惊讶的是，这个阶段的计算效率很高，使用单个 A100 只需大约 7 分钟即可完成。

环境搭建
----

基础环境配置如下：

*   **操作系统:**\*\* \*\***Ubuntu 18.04**
*   **CPUs:**\*\* \*\***单个节点具有 384GB 内存的 Intel CPU，物理CPU个数为2，每颗CPU核数为20**
*   **GPUs:**\*\* \*\***4 卡 A800 80GB GPUs**
*   **Python:**\*\* \*\***3.10 (需要先升级OpenSSL到1.1.1t版本（点击下载OpenSSL），然后再编译安装Python)，点击下载Python**
*   **NVIDIA驱动程序版本:**\*\* \*\***525.105.17，根据不同型号选择不同的驱动程序，点击下载。**
*   **CUDA工具包:**\*\* \*\***11.6，点击下载**
*   **cuDNN:**\*\* \*\***8.8.1.3\_cuda11，点击下载**

本文选择使用Doker镜像进行环境搭建。

首先，下载对应版本的Pytorch镜像。

```text-plain
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
```

镜像下载完成之后，创建容器。

```text-plain
docker run -dt --name minigpt4_env_dev --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/gdong/workspace/code:/workspace/code \
-v /home/gdong/workspace/data:/workspace/data \
-v /home/gdong/workspace/model:/workspace/model \
-v /home/gdong/workspace/output:/workspace/output \
-v /home/gdong/workspace/package:/workspace/package \
-w /workspace \
pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel \
/bin/bash
```

进入容器。

```text-plain
docker exec -it minigpt4_env_dev bash
```

安装 cv2 的依赖项。

```text-plain
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

安装其他依赖包。

```text-plain
pip install -r requirements.txt
```

其中，`requirements.txt`文件内容如下：

```text-plain
accelerate==0.16.0
aiohttp==3.8.4
aiosignal==1.3.1
async-timeout==4.0.2
attrs==22.2.0
bitsandbytes==0.37.0
cchardet==2.1.7
chardet==5.1.0
contourpy==1.0.7
cycler==0.11.0
filelock==3.9.0
fonttools==4.38.0
frozenlist==1.3.3
huggingface-hub==0.13.4
importlib-resources==5.12.0
kiwisolver==1.4.4
matplotlib==3.7.0
multidict==6.0.4
openai==0.27.0
packaging==23.0
psutil==5.9.4
pycocotools==2.0.6
pyparsing==3.0.9
python-dateutil==2.8.2
pyyaml==6.0
regex==2022.10.31
tokenizers==0.13.2
tqdm==4.64.1
transformers==4.28.0
timm==0.6.13
spacy==3.5.1
webdataset==0.2.48
scikit-learn==1.2.2
scipy==1.10.1
yarl==1.8.2
zipp==3.14.0
omegaconf==2.3.0
opencv-python==4.7.0.72
iopath==0.1.10
decord==0.6.0
tenacity==8.2.2
peft
pycocoevalcap
sentence-transformers
umap-learn
notebook
gradio==3.24.1
gradio-client==0.0.8
wandb
```

接下来，安装img2dataset库，用于后续下载数据集使用。

```text-plain
pip install img2dataset -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
```

数据集、模型权重及训练推理代码下载
-----------------

### 下载模型训练及推理代码

```text-plain
# commit id: 22d8888ca2cf0aac862f537e7d22ef5830036808
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
```

### 模型权重准备

预先准备好 Vicuna 权重，详情请查看官方文档。

在之前的文章**大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼**中，有讲解过如何合并Vicuna模型权重，在这里我直接使用之前合并好的Vicuna权重文件。

准备好 Vicuna 权重之后，在模型配置文件 `minigpt4.yaml` 中的第 16 行设置 Vicuna 权重的路径。

```text-plain
model:
  arch:mini_gpt4

  # vit encoder
  image_size:224
  drop_path_rate:0
  use_grad_checkpoint:False
  vit_precision:"fp16"
  freeze_vit:True
  freeze_qformer:True

  # Q-Former
  num_query_token:32

  # Vicuna
  llama_model:"/workspace/model/vicuna-7b-all-v1.1"

  # generation configs
  prompt:""

preprocess:
    vis_processor:
        train:
          name:"blip2_image_train"
          image_size:224
        eval:
          name:"blip2_image_eval"
          image_size:224
    text_processor:
        train:
          name:"blip_caption"
        eval:
          name:"blip_caption"
```

然后，下载预训练的 MiniGPT-4 检查点（checkpoint），用于模型推理。下载地址：与 Vicuna 7B 对齐的checkpoint(prerained\_minigpt4\_7b.pth) 或与 Vicuna 7B 对齐的checkpoint(pretrained\_minigpt4\_13b.pth)

如果服务器无法访问外网，需要预先下载好 VIT(eva\_vit\_g.pth)、Q-Former (blip2\_pretrained\_flant5xxl.pth)的权重以及Bert(bert-base-uncased)的Tokenizer。**如果服务器可以访问外网且网络状况良好，可以直接忽略以下步骤。**

eva\_vit\_g.pth和blip2\_pretrained\_flant5xxl.pth下载好之后，格式如下：

```text-plain
> ls -al hub/checkpoints/ --block-size=K
total 2401124K
drwxr-xr-x 2 root root       4K May  5 02:09 .
drwxr-xr-x 3 root root       4K May  7 02:34 ..
-rw------- 1 root root  423322K May  5 02:09 blip2_pretrained_flant5xxl.pth
-rw------- 1 root root 1977783K May  5 02:08 eva_vit_g.pth
```

同时需要设置环境变量：

```text-plain
# export TORCH_HOME=/workspace/model/cache/torch
export TORCH_HOME=/root/.cache/torch
```

bert-base-uncased下载好之后，格式如下：

```text-plain
> ls -al bert-base-uncased --block-size=K                
total 244K
drwxr-xr-x 2 root root   4K May  7 09:03 .
drwxrwxrwx 9 root root   4K May  7 09:02 ..
-rw-r--r-- 1 root root   1K May  7 09:03 config.json
-rw-r--r-- 1 root root   1K May  7 09:03 tokenizer_config.json
-rw-r--r-- 1 root root 227K May  7 09:03 vocab.txt
```

同时，需要修改`/workspace/code/MiniGPT-4/minigpt4/models/blip2.py`文件，改为本地加载Tokenizer：

```text-plain
class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        # TODO
        [[tokenizer]] = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("/workspace/model/bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    ...

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        # TODO
        [[encoder_config]] = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config = BertConfig.from_pretrained("/workspace/model/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
```

### 数据集准备

下面准备数据集，MiniGPT-4 的训练包含两个阶段，每个阶段使用的数据集不一样。

首先，准备第一阶段数据集。

| **图片来源** | **通过ViT-L过滤后的合成字幕** |
| --- | --- |
| CC3M+CC12M+SBU | Download |
| LAION115M | Download |

下载`ccs_synthetic_filtered_large.json`和`laion_synthetic_filtered_large.json`文件，并移动到对应的目录。

```text-plain
export MINIGPT4_DATASET=/workspace/data/blip
mkdir${MINIGPT4_DATASET}/cc_sbu
mkdir${MINIGPT4_DATASET}/laion
mv ccs_synthetic_filtered_large.json${MINIGPT4_DATASET}/cc_sbu
mv laion_synthetic_filtered_large.json${MINIGPT4_DATASET}/laion
```

进入MiniGPT-4项目的dataset目录，并拷贝转换数据格式和下载数据集的脚本。

```text-plain
cd dataset/
cp convert_cc_sbu.py${MINIGPT4_DATASET}/cc_sbu
cp download_cc_sbu.sh${MINIGPT4_DATASET}/cc_sbu

cp convert_laion.py${MINIGPT4_DATASET}/laion
cp download_laion.sh${MINIGPT4_DATASET}/laion
```

由于数据集太大，进入`${MINIGPT4_DATASET}/cc_sbu`和`${MINIGPT4_DATASET}/laion`文件夹，修改`convert_cc_sbu.py`和`convert_laion.py`脚本，改为仅下载一部分数据。

```text-plain
[[rows]] = [x.values() for x in data]

rows = []

for i, x in enumerate(data):
    if i >= 1000:
        break
    rows.append(x.values())
```

然后，将laion和cc\_sbu标注文件格式转换为img2dataset格式。

```text-plain
cd${MINIGPT4_DATASET}/cc_sbu
python convert_cc_sbu.py

cd${MINIGPT4_DATASET}/laion
python convert_laion.py
```

进入`${MINIGPT4_DATASET}/cc_sbu`和`${MINIGPT4_DATASET}/laion`文件夹，修改下载数据集脚本`download_cc_sbu.sh`和`download_laion.sh`，将`--enable_wandb`配置项改为`False`。

然后，执行脚本，使用img2dataset下载数据集。

```text-plain
cd${MINIGPT4_DATASET}/cc_sbu
sh download_cc_sbu.sh

cd${MINIGPT4_DATASET}/laion
sh download_laion.sh
```

下载完成之后的最终数据集结构如下所示：

```text-plain
> tree
.
|-- cc_sbu
|   |-- cc_sbu_dataset
|   |   |-- 00000.parquet
|   |   |-- 00000.tar
|   |   `-- 00000_stats.json
|   |-- ccs_synthetic_filtered_large.json
|   |-- ccs_synthetic_filtered_large.tsv
|   |-- convert_cc_sbu.py
|   `-- download_cc_sbu.sh
`-- laion
    |-- convert_laion.py
    |-- download_laion.sh
    |-- laion_dataset
    |   |-- 00000.parquet
    |   |-- 00000.tar
    |   `-- 00000_stats.json
    |-- laion_synthetic_filtered_large.json
    `-- laion_synthetic_filtered_large.tsv

4 directories, 14 files
```

之后，修改数据集配置文件。

修改配置文件`minigpt4/configs/datasets/laion/defaults.yaml`的第五行设置LAION数据集加载路径，具体如下所示：

```text-plain
datasets:
  laion:
    data_type:images
    build_info:
      storage:/workspace/data/blip/laion/laion_dataset/00000.tar
```

修改配置文件`minigpt4/configs/datasets/cc_sbu/defaults.yaml`的第五行设置 Conceptual Captoin 和 SBU 数据集加载路径，具体如下所示：

```text-plain
datasets:
  cc_sbu:
    data_type:images
    build_info:
      storage:/workspace/data/blip/cc_sbu/cc_sbu_dataset/00000.tar
```

接下来，准备第二阶段数据集，具体在此处下载，数据集文件夹结构如下所示。

```text-plain
cc_sbu_align
├── filter_cap.json
└── image
    ├── 2.jpg
    ├── 3.jpg
    ...
```

下载完成之后，在数据集配置文件`minigpt4/configs/datasets/cc_sbu/align.yaml`中的第 5 行设置数据集路径。

```text-plain
datasets:
  cc_sbu_align:
    data_type:images
    build_info:
      storage:/workspace/data/cc_sbu_align/
```

代码结构
----

MiniGPT-4 项目基于 BLIP2、Lavis 和 Vicuna 进行构建，使用 OmegaConf 基于 YAML 进行分层系统配置，整个代码结构如下所示：

```text-plain
.
|-- LICENSE.md
|-- LICENSE_Lavis.md
|-- MiniGPT_4.pdf
|-- PrepareVicuna.md
|-- README.md
|-- dataset # 数据集预处理
|   |-- README_1_STAGE.md
|   |-- README_2_STAGE.md
|   |-- convert_cc_sbu.py # 转换标注数据格式
|   |-- convert_laion.py
|   |-- download_cc_sbu.sh # 下载数据集
|   `-- download_laion.sh
|-- demo.py    # 模型测试/推理
|-- environment.yml
|-- eval_configs # 模型评估配置文件
|   `-- minigpt4_eval.yaml
|-- minigpt4
|   |-- __init__.py
|   |-- common
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- dist_utils.py # 模型权重缓存文件路径
|   |   |-- gradcam.py
|   |   |-- logger.py
|   |   |-- optims.py
|   |   |-- registry.py
|   |   `-- utils.py
|   |-- configs 
|   |   |-- datasets # 数据集配置文件
|   |   |   |-- cc_sbu
|   |   |   |   |-- align.yaml # cc_sbu对齐数据集配置文件
|   |   |   |   `-- defaults.yaml # cc_sbu数据集配置文件
|   |   |   `-- laion
|   |   |       `-- defaults.yaml # laion数据集配置文件
|   |   |-- default.yaml
|   |   `-- models # 模型配置文件
|   |       `-- minigpt4.yaml
|   |-- conversation
|   |   |-- __init__.py
|   |   `-- conversation.py
|   |-- datasets
|   |   |-- __init__.py
|   |   |-- builders
|   |   |   |-- __init__.py
|   |   |   |-- base_dataset_builder.py
|   |   |   `-- image_text_pair_builder.py
|   |   |-- data_utils.py
|   |   `-- datasets
|   |       |-- __init__.py
|   |       |-- base_dataset.py
|   |       |-- caption_datasets.py
|   |       |-- cc_sbu_dataset.py
|   |       |-- dataloader_utils.py
|   |       `-- laion_dataset.py
|   |-- models
|   |   |-- Qformer.py
|   |   |-- __init__.py
|   |   |-- base_model.py
|   |   |-- blip2.py # 初始化Bert Tokenizer 和 Qformer等
|   |   |-- blip2_outputs.py
|   |   |-- eva_vit.py
|   |   |-- mini_gpt4.py
|   |   `-- modeling_llama.py
|   |-- processors
|   |   |-- __init__.py
|   |   |-- base_processor.py
|   |   |-- blip_processors.py
|   |   `-- randaugment.py
|   |-- runners
|   |   |-- __init__.py
|   |   `-- runner_base.py
|   `-- tasks 
|       |-- __init__.py
|       |-- base_task.py 
|       `-- image_text_pretrain.py
|-- prompts 
|   `-- alignment.txt
|-- train.py # 模型训练
`-- train_configs # 模型训练配置文件
    |-- minigpt4_stage1_pretrain.yaml # 第一阶段预训练配置
    `-- minigpt4_stage2_finetune.yaml # 第二阶段微调配置
```

模型推理
----

首先，在评估配置文件`eval_configs/minigpt4_eval.yaml`中的第 11 行设置预训练checkpoint的路径(即刚刚下载的预训练的 MiniGPT-4 检查点)。

```text-plain
model:
  arch:mini_gpt4
  model_type:pretrain_vicuna
  freeze_vit:True
  freeze_qformer:True
  max_txt_len:160
  end_sym:"###"
  low_resource:True
  prompt_path:"prompts/alignment.txt"
  prompt_template:'###Human: {} ###Assistant: '
  ckpt:'/workspace/model/minigpt/prerained_minigpt4_7b.pth'

datasets:
  cc_sbu_align:
    vis_processor:
      train:
        name:"blip2_image_eval"
        image_size:224
    text_processor:
      train:
        name:"blip_caption"

run:
  task:image_text_pretrain
```

执行如下命令启动模型推理服务：

```text-plain
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

为了节省 GPU 内存，Vicuna 默认以 8 bit 进行加载，beam search 宽度为 1。此配置对于 Vicuna-13B 需要大约 23G GPU 内存、对于 Vicuna-7B 需要大约 11.5G GPU 内存。 如果你有更强大的 GPU，您可以通过在配置文件 `minigpt4_eval.yaml` 中将 `low_resource` 设置为 `False` 以 16 bit运行模型并使用更大的beam search宽度。

运行过程：

```text-plain
> python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
Initializing Chat
Loading VIT
Loading VIT Done
Loading Q-Former
Loading Q-Former Done
Loading LLAMA

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████| 2/2 [01:02<00:00, 31.47s/it]
Loading LLAMA Done
Load 4 training prompts
Prompt Example 
###Human: <Img><ImageHere></Img> Take a look at this image and describe what you notice. ###Assistant: 
Load BLIP2-LLM Checkpoint: /workspace/model/minigpt/prerained_minigpt4_7b.pth
Initialization Finished
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://71e239f43b078ebe0b.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces
```

模型推理测试：

![](4_大杀器，多模态大模型MiniGPT-4入坑指南_image.)

image.png

![](1_大杀器，多模态大模型MiniGPT-4入坑指南_image.)

image.png

模型训练
----

MiniGPT-4 的训练包含两个对齐阶段。

### 第一阶段：预训练

在预训练阶段，模型使用来自 Laion 和 CC 数据集的图像文本对进行训练，以对齐视觉和语言模型。

第一阶段之后，视觉特征被映射，可以被语言模型理解。 MiniGPT-4 官方在实验时使用了 4 个 A100。 除此之外，您还可以在配置文件 `train_configs/minigpt4_stage1_pretrain.yaml` 中更改保存路径，具体内容如下：

```text-plain
model:
  arch:mini_gpt4
  model_type:pretrain_vicuna
  freeze_vit:True
  freeze_qformer:True


datasets:
  laion:
    vis_processor:
      train:
        name:"blip2_image_train"
        image_size:224
    text_processor:
      train:
        name:"blip_caption"
    sample_ratio:115
  cc_sbu:
    vis_processor:
        train:
          name:"blip2_image_train"
          image_size:224
    text_processor:
        train:
          name:"blip_caption"
    sample_ratio:14


run:
  task:image_text_pretrain
  # optimizer
  lr_sched:"linear_warmup_cosine_lr"
  init_lr:1e-4
  min_lr:8e-5
  warmup_lr:1e-6

  weight_decay:0.05
  max_epoch:3
  batch_size_train:16
  batch_size_eval:2
  num_workers:4
  warmup_steps:500
  iters_per_epoch:500

  seed:42
  output_dir:"/workspace/output/minigpt4_stage1_pretrain"

  amp:True
  resume_ckpt_path:null

  evaluate:False
  train_splits:["train"]

  device:"cuda"
  world_size:1
  dist_url:"env://"
  distributed:True
```

接下来，通过以下命令启动第一阶段训练。

```text-plain
CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

运行过程：

```text-plain
> CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
| distributed init (rank 1, world 4): env://
| distributed init (rank 0, world 4): env://
| distributed init (rank 2, world 4): env://
| distributed init (rank 3, world 4): env://
2023-05-07 11:36:36,497 [INFO] 
=====  Running Parameters    =====
2023-05-07 11:36:36,498 [INFO] {
    "amp": true,
    "batch_size_eval": 2,
    "batch_size_train": 16,
    "device": "cuda",
    "dist_backend": "nccl",
    "dist_url": "env://",
    "distributed": true,
    "evaluate": false,
    "gpu": 0,
    "init_lr": 0.0001,
    "iters_per_epoch": 500,
    "lr_sched": "linear_warmup_cosine_lr",
    "max_epoch": 3,
    "min_lr": 8e-05,
    "num_workers": 4,
    "output_dir": "/workspace/output/minigpt4_stage1_pretrain",
    "rank": 0,
    "resume_ckpt_path": null,
    "seed": 42,
    "task": "image_text_pretrain",
    "train_splits": [
        "train"
    ],
    "warmup_lr": 1e-06,
    "warmup_steps": 500,
    "weight_decay": 0.05,
    "world_size": 4
}
2023-05-07 11:36:36,498 [INFO] 
======  Dataset Attributes  ======
2023-05-07 11:36:36,498 [INFO] 
======== laion =======
2023-05-07 11:36:36,499 [INFO] {
    "build_info": {
        "storage": "/workspace/data/blip/laion/laion_dataset/00000.tar"
    },
    "data_type": "images",
    "sample_ratio": 115,
    "text_processor": {
        "train": {
            "name": "blip_caption"
        }
    },
    "vis_processor": {
        "train": {
            "image_size": 224,
            "name": "blip2_image_train"
        }
    }
}
2023-05-07 11:36:36,499 [INFO] 
======== cc_sbu =======
2023-05-07 11:36:36,499 [INFO] {
    "build_info": {
        "storage": "/workspace/data/blip/cc_sbu/cc_sbu_dataset/00000.tar"
    },
    "data_type": "images",
    "sample_ratio": 14,
    "text_processor": {
        "train": {
            "name": "blip_caption"
        }
    },
    "vis_processor": {
        "train": {
            "image_size": 224,
            "name": "blip2_image_train"
        }
    }
}
2023-05-07 11:36:36,499 [INFO] 
======  Model Attributes  ======
2023-05-07 11:36:36,500 [INFO] {
    "arch": "mini_gpt4",
    "drop_path_rate": 0,
    "freeze_qformer": true,
    "freeze_vit": true,
    "image_size": 224,
    "llama_model": "/workspace/model/vicuna-7b-all-v1.1",
    "model_type": "pretrain_vicuna",
    "num_query_token": 32,
    "prompt": "",
    "use_grad_checkpoint": false,
    "vit_precision": "fp16"
}
2023-05-07 11:36:36,501 [INFO] Building datasets...
2023-05-07 11:36:36,503 [INFO] Building datasets...
Loading VIT
2023-05-07 11:36:58,812 [INFO] Downloading: "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth" to /root/.cache/torch/hub/checkpoints/eva_vit_g.pth

100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1.89G/1.89G [02:33<00:00, 13.2MB/s]
cache_file_path: /root/.cache/torch/hub/checkpoints/eva_vit_g.pth
2023-05-07 11:39:41,878 [INFO] freeze vision encoder
Loading VIT Done
Loading Q-Former
2023-05-07 11:39:45,781 [INFO] Downloading: "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth" to /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 413M/413M [00:31<00:00, 13.8MB/s]
cache_file_path: /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth
2023-05-07 11:40:18,140 [INFO] load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
2023-05-07 11:40:18,155 [INFO] freeze Qformer
Loading Q-Former Done
Loading LLAMA
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.79s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.94s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:16<00:00,  8.21s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:16<00:00,  8.13s/it]
Loading LLAMA Done
2023-05-07 11:43:21,365 [INFO] Start training
2023-05-07 11:43:28,791 [INFO] dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline).
2023-05-07 11:43:28,791 [INFO] Loaded 0 records for train split from the dataset.
module.llama_proj.weight
module.llama_proj.bias
2023-05-07 11:43:30,005 [INFO] number of trainable parameters: 3149824
2023-05-07 11:43:30,008 [INFO] Start training epoch 0, 500 iters per inner epoch.
Train: data epoch: [0]  [  0/500]  eta: 0:35:50  lr: 0.000001  loss: 7.4586  time: 4.3018  data: 0.0000  max mem: 20913
2023-05-07 11:43:34,313 [INFO] Reducer buckets have been rebuilt in this iteration.
Train: data epoch: [0]  [ 50/500]  eta: 0:03:04  lr: 0.000011  loss: 4.9250  time: 0.3323  data: 0.0000  max mem: 22076
Train: data epoch: [0]  [100/500]  eta: 0:02:29  lr: 0.000021  loss: 3.6569  time: 0.3376  data: 0.0000  max mem: 22076
Train: data epoch: [0]  [150/500]  eta: 0:02:06  lr: 0.000031  loss: 2.8653  time: 0.3415  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [200/500]  eta: 0:01:47  lr: 0.000041  loss: 2.5771  time: 0.3417  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [250/500]  eta: 0:01:28  lr: 0.000051  loss: 3.0763  time: 0.3375  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [300/500]  eta: 0:01:10  lr: 0.000060  loss: 2.3269  time: 0.3369  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [350/500]  eta: 0:00:52  lr: 0.000070  loss: 2.5431  time: 0.3403  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [400/500]  eta: 0:00:34  lr: 0.000080  loss: 2.6711  time: 0.3383  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [450/500]  eta: 0:00:17  lr: 0.000090  loss: 2.3690  time: 0.3426  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [499/500]  eta: 0:00:00  lr: 0.000100  loss: 1.5752  time: 0.3424  data: 0.0000  max mem: 22193
Train: data epoch: [0] Total time: 0:02:53 (0.3466 s / it)
2023-05-07 11:46:23,294 [INFO] Averaged stats: lr: 0.0001  loss: 3.2105
2023-05-07 11:46:23,297 [INFO] No validation splits found.
2023-05-07 11:46:23,334 [INFO] Saving checkpoint at epoch 0 to /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_0.pth.
2023-05-07 11:46:23,402 [INFO] Start training
2023-05-07 11:46:23,443 [INFO] Start training epoch 1, 500 iters per inner epoch.
Train: data epoch: [1]  [  0/500]  eta: 0:03:00  lr: 0.000095  loss: 1.9775  time: 0.3606  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [ 50/500]  eta: 0:02:34  lr: 0.000094  loss: 1.3029  time: 0.3486  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [100/500]  eta: 0:02:16  lr: 0.000093  loss: 1.1404  time: 0.3374  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [150/500]  eta: 0:01:59  lr: 0.000092  loss: 0.8192  time: 0.3376  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [200/500]  eta: 0:01:42  lr: 0.000091  loss: 0.4934  time: 0.3415  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [250/500]  eta: 0:01:25  lr: 0.000090  loss: 0.4390  time: 0.3402  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [300/500]  eta: 0:01:08  lr: 0.000089  loss: 0.2317  time: 0.3421  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [350/500]  eta: 0:00:51  lr: 0.000088  loss: 0.1960  time: 0.3413  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [400/500]  eta: 0:00:34  lr: 0.000087  loss: 2.0755  time: 0.3420  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [450/500]  eta: 0:00:17  lr: 0.000086  loss: 0.0773  time: 0.3405  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [499/500]  eta: 0:00:00  lr: 0.000085  loss: 0.1692  time: 0.3387  data: 0.0000  max mem: 22193
Train: data epoch: [1] Total time: 0:02:50 (0.3404 s / it)
2023-05-07 11:49:13,623 [INFO] Averaged stats: lr: 0.0001  loss: 0.7745
2023-05-07 11:49:13,625 [INFO] No validation splits found.
2023-05-07 11:49:13,660 [INFO] Saving checkpoint at epoch 1 to /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_1.pth.
2023-05-07 11:49:13,722 [INFO] Start training
2023-05-07 11:49:13,763 [INFO] Start training epoch 2, 500 iters per inner epoch.
Train: data epoch: [2]  [  0/500]  eta: 0:03:00  lr: 0.000085  loss: 0.2226  time: 0.3614  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [ 50/500]  eta: 0:02:34  lr: 0.000084  loss: 0.1156  time: 0.3454  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [100/500]  eta: 0:02:16  lr: 0.000083  loss: 0.0512  time: 0.3396  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [150/500]  eta: 0:01:59  lr: 0.000083  loss: 0.1134  time: 0.3421  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [200/500]  eta: 0:01:42  lr: 0.000082  loss: 0.0489  time: 0.3412  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [250/500]  eta: 0:01:25  lr: 0.000081  loss: 0.0693  time: 0.3409  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [300/500]  eta: 0:01:08  lr: 0.000081  loss: 0.0316  time: 0.3433  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [350/500]  eta: 0:00:51  lr: 0.000080  loss: 0.0372  time: 0.3464  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [400/500]  eta: 0:00:34  lr: 0.000080  loss: 0.0404  time: 0.3386  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [450/500]  eta: 0:00:17  lr: 0.000080  loss: 0.0523  time: 0.3396  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [499/500]  eta: 0:00:00  lr: 0.000080  loss: 0.0471  time: 0.3378  data: 0.0000  max mem: 22193
Train: data epoch: [2] Total time: 0:02:50 (0.3402 s / it)
2023-05-07 11:52:03,847 [INFO] Averaged stats: lr: 0.0001  loss: 0.2326
2023-05-07 11:52:03,849 [INFO] No validation splits found.
2023-05-07 11:52:03,885 [INFO] Saving checkpoint at epoch 2 to /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth.
2023-05-07 11:52:03,946 [INFO] No validation splits found.
2023-05-07 11:52:03,946 [INFO] Training time 0:08:42
```

**显存占用**：

```text-plain
Sun May  7 19:48:54 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   68C    P0   297W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  Off  | 00000000:5E:00.0 Off |                    0 |
| N/A   65C    P0   322W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  Off  | 00000000:AF:00.0 Off |                    0 |
| N/A   69C    P0   218W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  Off  | 00000000:D8:00.0 Off |                    0 |
| N/A   69C    P0   335W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11425      C   /opt/conda/bin/python           32436MiB |
|    1   N/A  N/A     11426      C   /opt/conda/bin/python           32436MiB |
|    2   N/A  N/A     11427      C   /opt/conda/bin/python           32436MiB |
|    3   N/A  N/A     11428      C   /opt/conda/bin/python           32436MiB |
+-----------------------------------------------------------------------------+
```

模型权重输出：

```text-plain
> tree minigpt4_stage1_pretrain/
minigpt4_stage1_pretrain/
`-- 20230507113
    |-- checkpoint_0.pth
    |-- checkpoint_1.pth
    |-- checkpoint_2.pth
    |-- log.txt
    `-- result

2 directories, 4 files
```

你也可以直接下载只有第一阶段训练的 MiniGPT-4 的 checkpoint，具体下载地址：13B 或 7B。

与第二阶段之后的模型相比，第一阶段的checkpoint经常生成不完整和重复的句子。

![](大杀器，多模态大模型MiniGPT-4入坑指南_image.)

**吃果冻不吐果冻皮**

致力于分享AI前沿技术（如：LLM/MLOps/RAG/智能体）、AI工程落地实践、AI基建（如：算力、网络、存储）等。

165篇原创内容

公众号

### 第二阶段：微调

在第二阶段，我们使用自己创建的小型高质量图文对数据集并将其转换为对话格式以进一步对齐 MiniGPT-4。

要启动第二阶段对齐，需先在`train_configs/minigpt4_stage2_finetune.yaml` 中指定第一阶段训练的checkpoint文件的路径。 当然，您还可以自定义输出权重路径，具体文件如下所示。

```text-plain
model:
  arch:mini_gpt4
  model_type:pretrain_vicuna
  freeze_vit:True
  freeze_qformer:True
  max_txt_len:160
  end_sym:"###"
  prompt_path:"prompts/alignment.txt"
  prompt_template:'###Human: {} ###Assistant: '
  ckpt:'/workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth'


datasets:
  cc_sbu_align:
    vis_processor:
      train:
        name:"blip2_image_train"
        image_size:224
    text_processor:
      train:
        name:"blip_caption"

run:
  task:image_text_pretrain
  # optimizer
  lr_sched:"linear_warmup_cosine_lr"
  init_lr:3e-5
  min_lr:1e-5
  warmup_lr:1e-6

  weight_decay:0.05
  max_epoch:5
  iters_per_epoch:200
  batch_size_train:12
  batch_size_eval:12
  num_workers:4
  warmup_steps:200

  seed:42
  output_dir:"/workspace/output/minigpt4_stage2_finetune"

  amp:True
  resume_ckpt_path:null

  evaluate:False
  train_splits:["train"]

  device:"cuda"
  world_size:1
  dist_url:"env://"
  distributed:True
```

然后，第二阶段微调的运行命令如下所示。 MiniGPT-4官方在实验中，仅使用了 1 卡 A100。

```text-plain
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

运行过程：

```text-plain
> CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
| distributed init (rank 0, world 1): env://
2023-05-07 12:03:11,908 [INFO] 
=====  Running Parameters    =====
2023-05-07 12:03:11,909 [INFO] {
    "amp": true,
    "batch_size_eval": 12,
    "batch_size_train": 12,
    "device": "cuda",
    "dist_backend": "nccl",
    "dist_url": "env://",
    "distributed": true,
    "evaluate": false,
    "gpu": 0,
    "init_lr": 3e-05,
    "iters_per_epoch": 200,
    "lr_sched": "linear_warmup_cosine_lr",
    "max_epoch": 5,
    "min_lr": 1e-05,
    "num_workers": 4,
    "output_dir": "/workspace/output/minigpt4_stage2_finetune",
    "rank": 0,
    "resume_ckpt_path": null,
    "seed": 42,
    "task": "image_text_pretrain",
    "train_splits": [
        "train"
    ],
    "warmup_lr": 1e-06,
    "warmup_steps": 200,
    "weight_decay": 0.05,
    "world_size": 1
}
2023-05-07 12:03:11,909 [INFO] 
======  Dataset Attributes  ======
2023-05-07 12:03:11,909 [INFO] 
======== cc_sbu_align =======
2023-05-07 12:03:11,910 [INFO] {
    "build_info": {
        "storage": "/workspace/data/cc_sbu_align/"
    },
    "data_type": "images",
    "text_processor": {
        "train": {
            "name": "blip_caption"
        }
    },
    "vis_processor": {
        "train": {
            "image_size": 224,
            "name": "blip2_image_train"
        }
    }
}
2023-05-07 12:03:11,910 [INFO] 
======  Model Attributes  ======
2023-05-07 12:03:11,910 [INFO] {
    "arch": "mini_gpt4",
    "ckpt": "/workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth",
    "drop_path_rate": 0,
    "end_sym": "###",
    "freeze_qformer": true,
    "freeze_vit": true,
    "image_size": 224,
    "llama_model": "/workspace/model/vicuna-7b-all-v1.1",
    "max_txt_len": 160,
    "model_type": "pretrain_vicuna",
    "num_query_token": 32,
    "prompt": "",
    "prompt_path": "prompts/alignment.txt",
    "prompt_template": "###Human: {} ###Assistant: ",
    "use_grad_checkpoint": false,
    "vit_precision": "fp16"
}
2023-05-07 12:03:11,910 [INFO] Building datasets...
Loading VIT
cache_file_path: /root/.cache/torch/hub/checkpoints/eva_vit_g.pth
2023-05-07 12:03:37,018 [INFO] freeze vision encoder
Loading VIT Done
Loading Q-Former
cache_file_path: /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth
2023-05-07 12:03:40,903 [INFO] load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
2023-05-07 12:03:40,916 [INFO] freeze Qformer
Loading Q-Former Done
Loading LLAMA
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.13s/it]
Loading LLAMA Done
Load 4 training prompts
Prompt Example 
###Human: <Img><ImageHere></Img> Describe this image in detail. ###Assistant: 
Load BLIP2-LLM Checkpoint: /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth
2023-05-07 12:06:34,005 [INFO] Start training
2023-05-07 12:06:40,005 [INFO] dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline).
2023-05-07 12:06:40,005 [INFO] Loaded 3439 records for train split from the dataset.
module.llama_proj.weight
module.llama_proj.bias
2023-05-07 12:06:40,029 [INFO] number of trainable parameters: 3149824
2023-05-07 12:06:40,030 [INFO] Start training epoch 0, 200 iters per inner epoch.
Train: data epoch: [0]  [  0/200]  eta: 0:15:02  lr: 0.000001  loss: 1.6358  time: 4.5127  data: 0.0000  max mem: 35512
2023-05-07 12:06:44,545 [INFO] Reducer buckets have been rebuilt in this iteration.
Train: data epoch: [0]  [ 50/200]  eta: 0:01:47  lr: 0.000008  loss: 1.3364  time: 0.6420  data: 0.0000  max mem: 36093
Train: data epoch: [0]  [100/200]  eta: 0:01:07  lr: 0.000015  loss: 1.2098  time: 0.6466  data: 0.0000  max mem: 36093
Train: data epoch: [0]  [150/200]  eta: 0:00:33  lr: 0.000023  loss: 1.0652  time: 0.6472  data: 0.0000  max mem: 36093
Train: data epoch: [0]  [199/200]  eta: 0:00:00  lr: 0.000030  loss: 1.0278  time: 0.6460  data: 0.0000  max mem: 36093
Train: data epoch: [0] Total time: 0:02:12 (0.6627 s / it)
2023-05-07 12:08:52,563 [INFO] Averaged stats: lr: 0.0000  loss: 1.2121
2023-05-07 12:08:52,565 [INFO] No validation splits found.
2023-05-07 12:08:52,601 [INFO] Saving checkpoint at epoch 0 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_0.pth.
2023-05-07 12:08:52,668 [INFO] Start training
2023-05-07 12:08:52,708 [INFO] Start training epoch 1, 200 iters per inner epoch.
Train: data epoch: [1]  [  0/200]  eta: 0:02:14  lr: 0.000028  loss: 0.9808  time: 0.6744  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [ 50/200]  eta: 0:01:35  lr: 0.000027  loss: 0.9252  time: 0.6336  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [100/200]  eta: 0:01:07  lr: 0.000026  loss: 1.0419  time: 0.7971  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [150/200]  eta: 0:00:33  lr: 0.000025  loss: 1.0150  time: 0.6486  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [199/200]  eta: 0:00:00  lr: 0.000023  loss: 0.9695  time: 0.6472  data: 0.0000  max mem: 36093
Train: data epoch: [1] Total time: 0:02:11 (0.6576 s / it)
2023-05-07 12:11:04,223 [INFO] Averaged stats: lr: 0.0000  loss: 0.9785
2023-05-07 12:11:04,227 [INFO] No validation splits found.
2023-05-07 12:11:04,264 [INFO] Saving checkpoint at epoch 1 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_1.pth.
2023-05-07 12:11:04,332 [INFO] Start training
2023-05-07 12:11:04,370 [INFO] Start training epoch 2, 200 iters per inner epoch.
Train: data epoch: [2]  [  0/200]  eta: 0:02:13  lr: 0.000023  loss: 1.1459  time: 0.6684  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [ 50/200]  eta: 0:01:38  lr: 0.000022  loss: 1.0003  time: 0.6580  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [100/200]  eta: 0:01:04  lr: 0.000020  loss: 0.8605  time: 0.6367  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [150/200]  eta: 0:00:32  lr: 0.000018  loss: 0.8841  time: 0.6445  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [199/200]  eta: 0:00:00  lr: 0.000017  loss: 0.8462  time: 0.6380  data: 0.0000  max mem: 36093
Train: data epoch: [2] Total time: 0:02:11 (0.6588 s / it)
2023-05-07 12:13:16,139 [INFO] Averaged stats: lr: 0.0000  loss: 0.9272
2023-05-07 12:13:16,143 [INFO] No validation splits found.
2023-05-07 12:13:16,178 [INFO] Saving checkpoint at epoch 2 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_2.pth.
2023-05-07 12:13:16,247 [INFO] Start training
2023-05-07 12:13:16,286 [INFO] Start training epoch 3, 200 iters per inner epoch.
Train: data epoch: [3]  [  0/200]  eta: 0:02:14  lr: 0.000017  loss: 0.8447  time: 0.6750  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [ 50/200]  eta: 0:01:37  lr: 0.000015  loss: 0.9082  time: 0.6517  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [100/200]  eta: 0:01:04  lr: 0.000014  loss: 0.9476  time: 0.6380  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [150/200]  eta: 0:00:32  lr: 0.000013  loss: 0.8131  time: 0.6443  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [199/200]  eta: 0:00:00  lr: 0.000012  loss: 0.8718  time: 0.6550  data: 0.0000  max mem: 36093
Train: data epoch: [3] Total time: 0:02:09 (0.6460 s / it)
2023-05-07 12:15:25,492 [INFO] Averaged stats: lr: 0.0000  loss: 0.9053
2023-05-07 12:15:25,495 [INFO] No validation splits found.
2023-05-07 12:15:25,530 [INFO] Saving checkpoint at epoch 3 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_3.pth.
2023-05-07 12:15:25,592 [INFO] Start training
2023-05-07 12:15:25,631 [INFO] Start training epoch 4, 200 iters per inner epoch.
Train: data epoch: [4]  [  0/200]  eta: 0:01:56  lr: 0.000012  loss: 0.8907  time: 0.5827  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [ 50/200]  eta: 0:01:37  lr: 0.000011  loss: 1.0402  time: 0.6489  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [100/200]  eta: 0:01:07  lr: 0.000010  loss: 0.9383  time: 0.6434  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [150/200]  eta: 0:00:33  lr: 0.000010  loss: 1.0148  time: 0.6435  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [199/200]  eta: 0:00:00  lr: 0.000010  loss: 0.7553  time: 0.6397  data: 0.0000  max mem: 36093
Train: data epoch: [4] Total time: 0:02:11 (0.6594 s / it)
2023-05-07 12:17:37,503 [INFO] Averaged stats: lr: 0.0000  loss: 0.8906
2023-05-07 12:17:37,507 [INFO] No validation splits found.
2023-05-07 12:17:37,543 [INFO] Saving checkpoint at epoch 4 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_4.pth.
2023-05-07 12:17:37,612 [INFO] No validation splits found.
2023-05-07 12:17:37,612 [INFO] Training time 0:11:03
```

显存占用：

```text-plain
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   69C    P0   311W / 300W |  40041MiB / 81920MiB |     94%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     62283      C   /opt/conda/bin/python           40038MiB |
+-----------------------------------------------------------------------------+
```

模型权重输出：

```text-plain
> tree minigpt4_stage2_finetune/
minigpt4_stage2_finetune/
`-- 20230507120
    |-- checkpoint_0.pth
    |-- checkpoint_1.pth
    |-- checkpoint_2.pth
    |-- checkpoint_3.pth
    |-- checkpoint_4.pth
    |-- log.txt
    `-- result

2 directories, 6 files
```

经过第二阶段对齐之后，MiniGPT-4 能够连贯地和用户友好地讨论图像。

至此，整个模型训练过程结束。接下来进行对训练的模型进行评估。

模型评估
----

首先，在评估配置文件`eval_configs/minigpt4_eval.yaml`中的第 11 行设置待评估模型的checkpoint路径，同模型推理。

```text-plain
model:
  arch: mini_gpt4
  ...
  low_resource: True
  prompt_path: "prompts/alignment.txt"
  prompt_template: '###Human: {} ###Assistant: '
  ckpt: '/workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_4.pth'
...
```

执行如下命令启动模型推理服务进行评估：

```text-plain
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

如果出现`Could not create share link. Please check your internet connection or our status page: https://status.gradio.app`这个问题，通常是由于网络环境不稳定造成的。可修改`demo.py`文件如下的代码，使用**IP:端口**访问即可。

```text-plain
[[demo]].launch(share=True, enable_queue=True)
demo.launch(server_name='0.0.0.0', share=True, enable_queue=True)
```

运行过程：

```text-plain
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
Initializing Chat
Loading VIT
cache_file_path: /root/.cache/torch/hub/checkpoints/eva_vit_g.pth
Loading VIT Done
Loading Q-Former
cache_file_path: /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth
Loading Q-Former Done
Loading LLAMA

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:35<00:00, 17.65s/it]
Loading LLAMA Done
Load 4 training prompts
Prompt Example 
###Human: <Img><ImageHere></Img> Take a look at this image and describe what you notice. ###Assistant: 
Load BLIP2-LLM Checkpoint: /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_4.pth
Initialization Finished
Running on local URL:  http://0.0.0.0:7860
```

模型评估测试：

![](2_大杀器，多模态大模型MiniGPT-4入坑指南_image.)

image.png

结语
--

本文给大家分享了多模态大模型MiniGPT-4的基本原理及模型训练推理方法，希望能够给大家带来帮助。

**参考文档**：

*   **MiniGPT-4**
*   **MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models**
*   **First Stage Data Preparation：Download the filtered Conceptual Captions, SBU, LAION datasets**
*   **Second Stage Data Preparation**