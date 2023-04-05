# 微调 Alpaca与LLaMA模型: 在一个自定义的数据集上进行大模型训练

很高兴给大家介绍基于Alpaca的Lora微调教程。 在本教程当中, 我们将通过检测Tweets上比特币的情绪分析，来探索Alpaca LoRa的微调过程。

# 环境配置
[lpaca Lora仓库](https://github.com/tloen/alpaca-lora)提供了基于低秩适应(Lora)重现斯坦福Alpaca大模型效果的处理代码。包括一个效果类GPT-3(text-davinci-003)的指令模型。模型参数可以扩展到13b, 30b, 以及65b, 同时Hugging Face的[PEFT](https://github.com/huggingface/peft)以及Dettmers提供的[bitsandybytes](https://github.com/TimDettmers/bitsandbytes)被用于在大模型微调中的提效与降本。

我们将在一个特定数据集上对Alpaca Lora进行一次完整的微调，首先从数据准备开始，最后是我们对模型的训练。 本教程将会覆盖数据处理、模型训练、以及使用最普世的自然语言处理库比如Transformers和Hugging Face进行结果评估。此外我们也会通过使用Gradio来介绍模型的部署以及测试。

在开始教程之前, 首先需要安装依赖包, 在本文中用到的依赖包如下:
```python
pip install -U pip
pip install accelerate==0.18.0
pip install appdirs==1.4.4
pip install bitsandbytes==0.37.2
pip install datasets==2.10.1
pip install fire==0.5.0
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/transformers.git
pip install torch==2.0.0
pip install sentencepiece==0.1.97
pip install tensorboardX==2.6
pip install gradio==3.23.0
```

在安装好以上依赖之后, 即可开始我们本次的课程之旅了， 首先让我们来引入对应的依赖包
```python
import json
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM 
import os
import sys
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import fire
import torch
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from pylab import rcParams

# 设置是用GPU还是CPU, 如果是mac M1芯片可以尝试mps
device = "cuda" if torch.cuda.is_available() else "cpu"
```

# 数据
本文中我们使用的数据是BTC推特上的情绪分析[数据集](https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment), 在Kaggle网站上就可以下载到对应的数据集, 本数据集包含了50000+BTC相关的推文。为清洗这些数据, 本文中移除了所有'RT'开始以及包含链接的数据。OK, 首先我们来下载数据集。在Kaggle网站上，直接选择到对应的数据集，下载即可。 当然也可以使用命令来下载。 

![数据集下载](https://github.com/csunny/algorithm/blob/master/images/btc_tweet.png)
> !gdown 1xQ89cpZCnafsW5T3G3ZQWvR7q682t2BN

我们可以通过Pandas来加载CSV文件数据

```python
df = pd.read_csv("../../data/BTC_Tweets_Updated.csv")
df.head()
```
![head](https://github.com/csunny/algorithm/blob/master/images/data_set.png)


在数据集上, 处理之后差不多有1900条推文, 情绪标签通过数字来表示，-1表示消极情绪, 0表示中性情绪, 1表示积极情绪。首先看一下数据分布
```python3
def.sentiment.value_counts()
```
```
['positive']    22937
['neutral']     21932
['negative']     5983
Name: Sentiment, dtype: int64
```

```python
df.Sentiment.value_counts().plot(kind="bar")
```
![plot](https://github.com/csunny/algorithm/blob/master/images/sentiment_plot.png)


通过数据分布我们可以看出, 负面情绪的分布明显较低，在评估模型的效果时我们应该重点考虑。

# 构建JSON数据集
在原始的alpaca仓库中，用到的数据集是JSON文件，是一份包含instruction、input、以及output的数据列表。
接下来我们将数据转换为对应的json格式。 
```python
def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"
 
dataset_data = [
    {
        "instruction": "Detect the sentiment of the tweet.",
        "input": row_dict["tweet"],
        "output": sentiment_score_to_name(row_dict["sentiment"])
    }
    for row_dict in df.to_dict(orient="records")
]
dataset_data[0]
```
```json
{
  "instruction": "Detect the sentiment of the tweet.",
  "input": "@p0nd3ea Bitcoin wasn't built to live on exchanges.",
  "output": "Positive"
}
```

最后我们将数据保存到文件，用于之后的模型训练。 
```python
import json
with open("alpaca-bitcoin-sentiment-dataset.json", "w") as f:
   json.dump(dataset_data, f)
```

# 模型权重
虽然没有原始的LLaMA模型的权重可以使用, 但它们被泄漏了并且被改编为HuggingFace的模型库可以跟Transformers一起使用。在这里我们使用decapoda研究的权重。

```python
BASE_MODEL = "decapoda-research/llama-7b-hf"
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = (
    0
)
tokenizer.padding_side = "left"
```

这段使用LlamaFOrCausalLM类来加载预训练的Llama模型,LlamaFOrCausalLM类被HuggingFace的Transformers库所实现。  load_in_8bit=True参数使用8位量化加载模型以减少内存使用并提高推理速度。

同时以上代码也加载了分词器通过同样的Llama模型, 使用Transformers的LlamaTokenizer类， 并且设置了一些额外的属性比如pad_token_id设置为了0来表现未知的token, 设置了padding_side 设置为了left, 为了在左侧填充序列。

# 数据集
现在我们已经加载了模型和分词器, 我们可以通过HuggingFace提供的load_dataset()方法来处理我们之前保存的数据了。
```
data = load_dataset("json", data_files="alpaca-bitcoin-sentiment-dataset.json")
data["train]
```

```python
Dataset({
    features: ['instruction', 'input', 'output'],
    num_rows: 1897
})

```
接下来, 我们需要需要从数据集中构建提示词，并进行标记。
```python
def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt
```

上述第一个函数generate_prompt 从数据集里面取一个数据点并通过组合instruction、input、以及output值来生成一个提示。 第二个函数tokenize获取生成的提示词并对其进行分词。它也会给词追加一个结束序列并设置一个标签, 保持跟输入序列一致。第三个函数generate_and_tokenize_prompt 组合了第一个和第二个函数在一个步骤里面生成并且分词提示词。

数据准备的最后一步是将数据拆分为单独的训练集和验证集
```python
train_val = data["train"].train_test_split(
    test_size = 200, shuffer=True, seed=42
)

train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)

val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)

```
我们使用200条数据作为验证数据集并将数据打撒。generate_and_tokenize_prompt函数被用于数据的每一个样本来生成标记好的提示词。


# 训练
训练过程需要依赖几个参数, 这些参数主要来自原始库中的微调脚本
```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
 
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "experiments"

```
现在我们就可以准备模型来训练了
```
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

```

我们初始化并且准备好了模型来训练Lora算法, 这是一种量化形式，可以减少模型大小和内存使用量，而不会显著降低准确性。

LoraConfig 是一个LORA算法超参数的类，比如像正则化强度(lora_alpha),  丢弃概率(lora_dropout), 以及要压缩的目标模块(target_modules)

在训练过程中, 我们将使用来自HuggingFace的Transformers库中的Trainer类。
```python
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)
```

这段代码创建了一个TrainingArguments 训练参数对象, 其设定了一系列的变量以及超参数来训练基础模型，这些参数包括
- gradient_accumulation_steps: 梯度下降的步长, 是指在训练神经网络时，累积多个小 batch 的梯度更新权重参数的方法， 主要在反向传播跟梯度更新。

- warmup_steps: 优化器的预热步骤数。
- max_steps: 训练模型的的步数上限
- learning_rate: 优化器的学习率
- fp16: 使用 16 位精度进行训练

```python
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

```
DataCollatorForSeq2Seq 是来自Transformers库中的类, 创建批量的输入/输出给Seq2Seq模型。在本代码中, DataCollatorForSeq2Seq是通过下列参数实例化的对象
- pad_to_multiple_of: 表示最大序列长度的整数，四舍五入到最接近该值的倍数。
- padding: 一个布尔值，指示是否将序列填充到指定的最大长度

现在我们有所有的必要条件了，接下来我们就可以训练我们的模型了
```python
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collector=data_collector
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict=(
    lamba self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

trainer.train()
model.save_pretrained(OUTPUT_DIR)

```

# 推理



# 总结
综上所述, 我们已经成功的微调了Llama模型通过Alpaca-LoRa方法, 在一份BTC情绪分析数据集上。我们已经使用了HuggingFace提供的Transformers库和HuggingFace提供的数据集的库来加载并且处理数据，同时使用Transformer库来训练我们的模型，最后我们将模型部署在了HuggingFace的模型仓库当中, 并介绍了如何用Gradio应用来使用我们的模型。

# 附录
- [alpaca微调](https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning)
- [alpaca-lora](https://github.com/tloen/alpaca-lora/)
- [peft](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [BTC tweet数据集](https://www.kaggle.com/datasets/aisolutions353/btc-tweets-sentiment)
- [alpaca数据格式](https://github.com/tatsu-lab/stanford_alpaca#data-release)
- [llama模型](https://huggingface.co/decapoda-research/llama-7b-hf)
- [lora Config](https://github.com/huggingface/peft/blob/86f4e45dccf873dd04348b08dbadd30d50171ccc/src/peft/tuners/lora.py#L40)