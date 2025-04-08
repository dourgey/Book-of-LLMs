**第四部分：模型微调与对齐**

# 第6章：微调：让大模型适应特定任务

在第五章，我们见证了通过大规模预训练，Transformer 模型如何从海量文本中汲取通用语言知识和世界常识，化身为强大的“基础模型”。这些预训练模型（Pre-trained Models, PTMs）拥有了理解和生成语言的惊人潜力。然而，通用知识并不等同于特定领域的专长或完成具体任务的精确能力。就像一位博览群书的通才，虽然知识面广，但在解决某个专业问题（如写一份法律文书、诊断特定疾病的医学影像、或者进行特定产品的情感分析）时，可能不如经过专门训练的专家。

为了弥合通用能力与特定任务需求之间的差距，我们需要对预训练模型进行进一步的“塑造”，使其专注于我们想要解决的问题。这个过程就是**微调 (Fine-tuning)**。微调利用特定任务的标注数据，在预训练模型的基础上进行训练，调整模型参数，使其更好地适应新任务的要求。

本章将深入探讨微调的原理、方法和策略。我们将从最直接的全参数微调入手，通过案例学习如何将其应用于文本分类和生成任务。随后，我们将重点介绍参数高效微调（PEFT）技术，如 Adapter Tuning 和 LoRA，它们旨在以更低的成本实现模型适配。最后，我们还将探讨指令微调（Instruction Tuning）这一提升模型泛化和指令遵循能力的重要技术。

**6.1 为何需要微调？通用知识 vs. 特定任务**

预训练模型通过 MLM、CLM 等任务学习了语言的通用模式，例如：

* **语法结构:** 主谓宾关系、词性、句法依赖等。
* **语义信息:** 词语含义、近义词、反义词、上下文消歧。
* **世界知识:** 事实性信息（“巴黎是法国的首都”）、常识（“天空是蓝色的”）、概念关系等。
* **基础推理:** 简单的因果关系、模式识别。

这些通用能力使得预训练模型在**零样本 (Zero-shot)** 或**少样本 (Few-shot)** 场景下，仅通过**提示工程 (Prompt Engineering)**（在第8章详述）就能展现出一定的任务解决能力。例如，我们可以给 GPT 模型一个提示：“将以下英文翻译成法文：Hello world ->”，模型或许能直接给出 “Bonjour le monde”。或者给它几个情感分类的例子，它就能对新的句子进行分类。

然而，仅依赖提示工程（也称为上下文学习 In-context Learning）存在一些局限性：

1. **性能可能不稳定或次优:** 对于需要高精度或特定领域知识的任务，零样本/少样本的性能通常不如经过专门微调的模型。模型可能无法完全理解提示的细微差别，或者表现不稳定。
2. **对提示的设计敏感:** 模型输出的质量高度依赖于提示的措辞、示例的选择和格式，设计好的提示本身就需要技巧和实验。
3. **上下文长度限制:** 提示中能包含的示例数量受到模型最大上下文窗口的限制。
4. **难以注入深度领域知识:** 对于需要大量特定领域术语、规则或风格的任务（如医学报告生成、法律合同审查），仅靠提示难以让模型掌握所需的深度知识。
5. **控制性较弱:** 很难精确控制模型的输出风格、格式或确保其遵循特定的约束。

**微调 (Fine-tuning)** 提供了一种更直接、更有效的方式来将预训练模型的通用能力**适配 (Adapt)** 到特定任务上。通过在任务相关的标注数据上进一步训练模型（通常是更新全部或部分参数），我们可以：

* **注入任务特定知识:** 让模型学习特定领域的术语、模式和关系。
* **调整模型行为:** 使模型的输出更符合特定任务的要求（如特定的输出格式、语言风格）。
* **提升任务性能:** 在特定任务的评估指标上通常能取得比零/少样本提示更好的结果。
* **提高鲁棒性和一致性:** 使模型在任务上的表现更稳定可靠。

简而言之，如果预训练是打基础，那么微调就是在特定方向上进行**专业深造**。当我们需要模型在某个或某些任务上达到高水平、高可靠性的表现时，微调通常是必不可少的步骤。

**6.2 全参数微调（Full Fine-tuning）**

全参数微调是最早也是最直接的微调方法。顾名思义，它在微调阶段会**更新预训练模型的所有参数**。

**6.2.1 原理与流程**

全参数微调的基本流程如下：

1. **加载预训练模型:** 选择一个合适的预训练模型（如 BERT, RoBERTa, GPT-2, Llama）及其对应的分词器。模型的选择取决于目标任务（NLU 任务通常选 Encoder-Only 或 Encoder-Decoder，NLG 任务通常选 Decoder-Only 或 Encoder-Decoder）。
2. **添加/修改任务头 (Task Head):** 根据具体任务，在预训练模型的顶部添加一个或修改其原有的输出层（任务头）。例如：
   * **文本分类:** 在 [CLS] Token 对应的输出向量（对于BERT类模型）或最后一个 Token 的输出向量上添加一个线性层 + Softmax 来进行分类。
   * **序列标注 (NER):** 在每个 Token 的输出向量上添加一个线性层 + Softmax 来预测该 Token 的标签（如 B-PER, I-PER, O）。
   * **问答 (抽取式):** 添加两个线性层，分别预测答案在原文中的起始位置和结束位置。
   * **文本生成:** 通常可以直接使用预训练模型的语言模型头（LM Head），该头本身就是一个预测词汇表概率分布的线性层 + Softmax。
3. **准备标注数据:** 获取适用于目标任务的标注数据集（如带有情感标签的句子、带有 NER 标签的文本、问题-上下文-答案对、特定风格的文本等）。
4. **数据预处理:** 使用加载的分词器对标注数据进行编码，处理成模型接受的格式（包括 `input_ids`, `attention_mask`, `token_type_ids` (如果需要), 以及 `labels`）。
5. **设置优化器和学习率:**
   * 通常继续使用 AdamW 优化器。
   * 关键在于**学习率 (Learning Rate)**：微调时的学习率需要**远小于**预训练时的学习率。典型的微调学习率范围是 `1e-5` 到 `5e-5` 之间。因为预训练模型已经学到了很多有用的知识，我们只需要在原有基础上进行“微调”，过大的学习率可能会破坏预训练学到的知识，导致“灾难性遗忘 (Catastrophic Forgetting)”。
   * 学习率调度器（如带预热的线性衰减）仍然常用，但总步数通常远少于预训练。
6. **训练:** 在标注数据上进行训练，最小化任务特定的损失函数（如分类任务的交叉熵损失，生成任务的 CLM 损失）。训练通常只需要几个（如 3-5 个）Epoch 就能达到不错的效果。
7. **评估:** 在验证集上评估模型性能，选择效果最好的检查点。
8. **测试/部署:** 在测试集上进行最终评估，或将微调后的模型部署到应用中。

**6.2.2 任务特定输出层的添加**

如何添加任务头取决于模型架构和任务类型。Hugging Face `transformers` 库提供了极大的便利，它为许多常见的预训练模型内置了各种任务的专用类（如 `BertForSequenceClassification`, `GPT2ForSequenceClassification`, `BertForTokenClassification`, `BertForQuestionAnswering`, `GPT2LMHeadModel` 等）。这些类会自动在预训练模型的基础上添加相应的任务头。

例如，`BertForSequenceClassification` 会在 BERT 模型的基础上添加一个 Dropout 层和一个线性层，作用于 [CLS] Token 的输出向量，用于分类。

```python
from transformers import BertConfig, BertModel, BertForSequenceClassification
import torch.nn as nn

# 假设有一个预训练的 BERT 模型实例
config = BertConfig.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# --- 手动添加分类头 ---
class BertClassifierManual(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # BERT 输出维度是 config.hidden_size
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # outputs 是一个 BaseModelOutputWithPoolingAndCrossAttentions 对象
        # 我们通常使用 pooler_output，它是 [CLS] token 经过线性层+Tanh 的输出
        pooled_output = outputs.pooler_output # shape: (batch_size, hidden_size)
        # 或者直接取 [CLS] token 的 last_hidden_state
        # cls_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # shape: (batch_size, num_labels)
        return logits

num_classes = 2 # 例如，二分类
manual_classifier = BertClassifierManual(bert_model, num_classes)

# --- 使用 Hugging Face 内置类 ---
# 这个类内部已经包含了类似的 dropout 和 linear 层
hf_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# 两者结构类似，HF 内置类更方便使用
print("Manual Classifier Head:\n", manual_classifier.classifier)
print("\nHugging Face Classifier Head:\n", hf_classifier.classifier)
```

对于生成任务（如使用 GPT-2），通常不需要额外添加层，因为 `GPT2LMHeadModel` 本身就包含了一个语言模型头，可以直接用于预测下一个词的概率。微调过程就是在这个 LM 头上继续训练，使其生成的文本更符合目标数据集的风格或内容。

**6.2.3 案例：微调预训练BERT进行文本分类（如情感分析）**

**任务:** 对电影评论进行情感分类（正面/负面）。
**数据集:** IMDB 数据集（包含 5 万条电影评论，一半正面一半负面，训练集和测试集各半）。
**模型:** `bert-base-uncased`。

**代码 (使用 Hugging Face `transformers` Trainer API):**

```python
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. 定义模型名称和加载分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 2. 加载数据集
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

# 3. 数据预处理函数
def preprocess_function(examples):
    # 使用分词器编码文本，截断到最大长度 (BERT 通常是 512)
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print("Preprocessing dataset...")
# 对整个数据集应用预处理 (可以并行处理)
# batched=True 加速处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Hugging Face Trainer API 需要数据集包含 'labels' 列
# IMDB 数据集本身就有 'label' 列 (0 for neg, 1 for pos)
# 如果名字不同或需要转换，可以在这里处理
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 只取一小部分数据进行快速演示
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
print(f"Using {len(small_train_dataset)} training examples and {len(small_eval_dataset)} evaluation examples.")

# 4. 加载预训练模型 (带分类头)
print("Loading pre-trained model...")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # num_labels=2 for binary classification

# 5. 定义评估指标计算函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary') # for binary classification
    return {"accuracy": accuracy, "f1": f1}

# 6. 定义训练参数
output_dir = "./imdb_bert_finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,             # 简化训练轮数
    per_device_train_batch_size=8,  # 减小 batch size 以适应显存
    per_device_eval_batch_size=8,
    warmup_steps=100,               # 预热步数
    weight_decay=0.01,              # 权重衰减
    logging_dir='./logs',           # 日志目录
    logging_steps=50,               # 每 50 步记录一次日志
    evaluation_strategy="steps",    # 每 N 步评估一次
    eval_steps=100,                 # 每 100 步评估一次
    save_strategy="steps",          # 每 N 步保存一次模型
    save_steps=100,
    load_best_model_at_end=True,    # 训练结束后加载最佳模型
    metric_for_best_model="accuracy", # 以 accuracy 作为选择最佳模型的标准
    fp16=torch.cuda.is_available(), # 如果有 GPU，启用混合精度训练
    report_to="tensorboard",        # 将日志报告给 TensorBoard (需要安装 tensorboard)
    learning_rate=2e-5,             # 微调常用的学习率
)

# 7. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,            # 传递 tokenizer 以便正确处理 padding
    compute_metrics=compute_metrics,
)

# 8. 开始训练
print("Starting fine-tuning...")
trainer.train()

# 9. 评估最终模型
print("\nEvaluating final model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 10. 保存最终模型和分词器
print(f"\nSaving final model to {output_dir}")
trainer.save_model(output_dir)
# tokenizer.save_pretrained(output_dir) # Trainer 内部会自动保存 tokenizer (如果传递了)

# --- 如何使用微调后的模型进行预测 ---
# from transformers import pipeline
# pipe = pipeline("text-classification", model=output_dir, device=0 if torch.cuda.is_available() else -1)
# result = pipe("This movie was absolutely fantastic!")
# print(result) # [{'label': 'LABEL_1', 'score': 0.99...}] (LABEL_1 对应正面)
# result = pipe("What a waste of time, terrible acting.")
# print(result) # [{'label': 'LABEL_0', 'score': 0.99...}] (LABEL_0 对应负面)
```

这个例子展示了使用 Hugging Face `Trainer` API 进行全参数微调的标准流程。`Trainer` 封装了训练循环、评估、日志记录、检查点保存、混合精度、分布式训练（只需修改 `TrainingArguments`）等复杂细节，极大地简化了微调过程。

**6.2.4 案例：微调预训练GPT进行文本生成（特定风格或主题）**

**任务:** 让 GPT-2 模型生成莎士比亚风格的文本。
**数据集:** 可以使用 `datasets` 库中的 `tiny_shakespeare` 数据集。
**模型:** `gpt2`。

**代码 (使用 Hugging Face `transformers` Trainer API):**

```python
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import math

# 1. 配置参数
model_name = 'gpt2'
dataset_name = 'tiny_shakespeare'
output_dir = "./shakespeare_gpt2_finetuned"
block_size = 128   # 序列长度
batch_size = 8
learning_rate = 5e-5
epochs = 1
warmup_steps = 100
gradient_accumulation_steps = 2 # 模拟更大的 batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 加载分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
# 确保模型配置中的 pad_token_id 设置正确
model.config.pad_token_id = tokenizer.pad_token_id

# 3. 加载和预处理数据集
dataset = load_dataset(dataset_name)

def tokenize_function(examples):
    # 直接编码 "text" 字段
    output = tokenizer(examples["text"])
    # 为了 group_texts, 可能需要返回 list of lists 形式
    # tokenizer 返回的是 {'input_ids': [...], 'attention_mask': [...]}
    # Hugging Face 的 group_texts 期望 {'input_ids': [[...], [...]], 'attention_mask': [[...], [...]]} ?
    # 检查一下，datasets map 对 list of strings 的处理
    # 直接返回 tokenizer 输出即可
    return output

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 同样进行分块 (代码同上一个例子)
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print("Grouping texts into blocks...")
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# tiny_shakespeare 只有一个 'train' split，我们手动拆分一下
train_val_split = lm_datasets['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']
print(f"Using {len(train_dataset)} training blocks and {len(eval_dataset)} evaluation blocks.")


# 4. 数据整理器 (Data Collator)
# 对于语言模型任务，我们需要一个 Data Collator 来处理批次内的 padding
# DataCollatorForLanguageModeling 会自动创建 'labels'，并且可以处理 MLM 或 CLM
# 对于 CLM (GPT2)，它会复制 input_ids 作为 labels
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # mlm=False 表示 CLM

# 5. 定义训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss", # 对于 LM，通常监控 loss 或 perplexity
    greater_is_better=False,     # loss 越小越好
    fp16=torch.cuda.is_available(),
    report_to="tensorboard",
)

# 6. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator, # 使用 Data Collator
    # compute_metrics 可以用来计算 perplexity
)

# 定义 perplexity 计算
def compute_perplexity(eval_preds):
    # Trainer 的 evaluate 默认不返回 loss，需要修改
    # 或者直接在 evaluate 循环后计算
    # 这里先省略，只看 loss
    pass

# 7. 开始训练
print("Starting fine-tuning...")
trainer.train()

# 8. 评估最终模型 (Trainer 会自动计算 loss，也可以手动计算 perplexity)
print("\nEvaluating final model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
try:
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Perplexity: {perplexity:.2f}")
except OverflowError:
    perplexity = float("inf")
    print("Perplexity overflowed.")

# 9. 保存模型
print(f"\nSaving final model to {output_dir}")
trainer.save_model(output_dir)

# 10. 生成文本示例
print("\nGenerating Shakespeare-style text...")
model.eval() # 确保模型在评估模式
prompt = "To be, or not to be, that is the question:"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=100, # 生成更长一点
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8, # 可以调整温度控制随机性
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(f"Generated text:\n{generated_text}")
```

这个例子同样使用了 `Trainer` API，但针对的是生成任务（CLM）。关键区别在于使用了 `GPT2LMHeadModel` 和 `DataCollatorForLanguageModeling`。通过在莎士比亚文本上微调，模型生成的文本会逐渐模仿其独特的语言风格和词汇。

全参数微调虽然直观有效，但其主要缺点在于**成本高昂**：

* **存储成本:** 每微调一个新任务，就需要保存一份完整的、与原始预训练模型一样大的模型副本（数十亿甚至千亿参数）。任务数量多时，存储开销巨大。
* **计算成本:** 训练所有参数仍然需要相当的计算资源（尽管比预训练少得多），对于非常大的模型，微调本身也可能很慢。
* **部署挑战:** 如果有多个微调后的模型，在线上部署和切换这些庞然大物也很复杂。

这些挑战催生了**参数高效微调 (Parameter-Efficient Fine-tuning, PEFT)** 技术。

**6.3 参数高效微调（Parameter-Efficient Fine-tuning, PEFT）**

PEFT 的核心思想是在微调时**冻结 (Freeze)** 绝大部分预训练模型的参数（保持不变），只**引入并训练少量额外的参数**（通常只占总参数量的很小一部分，如 0.1% - 1%），或者只调整模型内部结构的一小部分。目标是：在尽可能**接近**全参数微调性能的同时，显著**降低**计算和存储成本。

**6.3.1 动机：降低微调成本和存储需求**

PEFT 主要解决以下痛点：

* **存储爆炸:** 不再需要为每个任务存储整个大模型，只需存储少量任务特定的参数（如 Adapter 权重、LoRA 矩阵、Prefix/Prompt 向量）。
* **训练加速:** 由于可训练参数大幅减少，训练速度通常更快，所需计算资源更少。
* **更容易部署:** 可以加载同一个基础大模型，然后根据需要动态加载或插入不同的 PEFT 模块，实现多任务服务。
* **缓解灾难性遗忘:** 由于大部分预训练参数被冻结，模型在学习新任务时，不太容易忘记在预训练阶段学到的通用知识。

PEFT 已经成为微调大模型的主流方法之一，特别是对于资源受限或者需要适配大量任务的场景。

**6.3.2 Adapter Tuning：原理与实现思路**

Adapter Tuning 是最早提出并获得成功的 PEFT 方法之一 (Houlsby et al., 2019)。

* **原理:** 在 Transformer 模型的**每一层**（或某些层）的**内部**，插入小型、瓶颈状的神经网络模块，称为**适配器 (Adapters)**。通常插入在两个子层（如 Self-Attention 和 FFN）之后，但在 Add & Norm 操作之前。
* **结构:** 一个典型的 Adapter 模块包含：

  1. 一个**下投影 (Down-project)** 线性层，将 `d_model` 维的输入投影到一个较小的**瓶颈维度 (Bottleneck Dimension)** `d_adapter`（例如 64, 128，远小于 `d_model`）。
  2. 一个**非线性激活函数** (如 ReLU, GeLU)。
  3. 一个**上投影 (Up-project)** 线性层，将 `d_adapter` 维的向量投影回 `d_model` 维。
  4. 一个**残差连接**：将 Adapter 的输出**加到**其输入上。
* **训练:** 在微调时，**冻结**原始 Transformer 模型的所有参数，**只训练**新插入的所有 Adapter 模块的参数。
* **实现思路:** 需要修改 Transformer 模型的 `forward` 代码，在适当的位置插入 Adapter 模块的调用。
* **图示 (Adapter 插入位置):**

  ```
  Input x -> SelfAttention -> + -> Norm -> Input x'
                            ^          |
                            | Adapter1 |
                            ------------

  Input x' -> FFN ---------> + -> Norm -> Output y
                            ^          |
                            | Adapter2 |
                            ------------
  ```
* **优点:** 参数效率高（只训练 Adapters），性能在很多任务上接近全参数微调。
* **缺点:** 增加了模型的**推理延迟**，因为需要在每一层都额外计算 Adapter 部分；修改模型结构相对复杂一些。

**6.3.3 LoRA（Low-Rank Adaptation）：原理与实现思路**

LoRA (Hu et al., 2021) 是目前最流行和广泛应用的 PEFT 方法之一。它基于一个观察：预训练模型通常具有很低的**本征维度 (Intrinsic Dimension)**，意味着模型参数的改变量（`ΔW = W_finetuned - W_pretrained`）可能位于一个低秩子空间。

* **原理:** LoRA 不直接学习 `ΔW`（维度很大），而是学习 `ΔW` 的一个**低秩分解 (Low-Rank Decomposition)**。具体来说，对于预训练模型中的某个权重矩阵 `W` (例如 Attention 中的 `W_q`, `W_k`, `W_v`, `W_o` 或 FFN 中的线性层权重)，LoRA 引入两个**小的、可训练的**矩阵 `A` 和 `B`，其中 `A` 的维度是 `d_model × r`，`B` 的维度是 `r × d_model`，而 `r` 是一个**非常小**的秩 (Rank)，称为 LoRA 的秩（如 4, 8, 16, 远小于 `d_model`）。在微调过程中，`W` 本身保持冻结，我们只训练 `A` 和 `B`。模型的前向传播计算变为 `h = (W + BA)x = Wx + BAx`。
* **训练:** 只训练所有 LoRA 矩阵 `A` 和 `B`，原始模型 `W` 冻结。可训练参数量大大减少（约为 `2 * d_model * r * num_lora_layers`）。
* **推理:**

  * **方式一（分离式）:** 保持 `A` 和 `B` 分离，推理时计算 `Wx + BAx`。会略微增加计算量。
  * **方式二（合并式）:** 在训练完成后，可以**计算出 `ΔW = BA` 并将其加到原始权重 `W` 上**，得到一个新的权重矩阵 `W' = W + BA`。然后就可以丢弃 `A` 和 `B`，直接使用 `W'` 进行推理。这种方式**完全不增加推理时的参数量和计算延迟**，非常适合部署。
* **应用位置:** LoRA 通常应用于 Transformer 中的 Attention 层的 `W_q` 和 `W_v` 矩阵（有时也用于 `W_k`, `W_o` 或 FFN 层），经验表明这样效果较好。
* **优点:**

  * 参数效率极高。
  * 性能强大，在很多任务上能媲美甚至超过全参数微调。
  * 推理时可以合并权重，**不增加延迟**。
  * 易于实现和切换任务（只需替换 LoRA 权重 `A` 和 `B`）。
* **缺点:** 秩 `r` 和应用 LoRA 的层需要作为超参数进行调整。
* **代码示例：使用 `peft` 库对预训练模型应用LoRA进行微调**

   Hugging Face 的 `peft` (Parameter-Efficient Fine-Tuning) 库极大地简化了 LoRA 等 PEFT 方法的应用。

```python
   import torch
   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
   from peft import get_peft_model, LoraConfig, TaskType # 导入 peft 相关库
   from sklearn.metrics import accuracy_score

   # 1. 配置参数
   model_name = "bert-base-uncased"
   dataset_name = "sst2" # GLUE 数据集中的情感分析任务 (句子级二分类)
   output_dir = "./sst2_bert_lora"
   lora_r = 8         # LoRA 秩
   lora_alpha = 16    # LoRA alpha (缩放因子, 常设为 r 的倍数)
   lora_dropout = 0.1
   # 应用 LoRA 的模块 (通常是 attention query 和 value)
   lora_target_modules = ["query", "value"]
   batch_size = 16
   learning_rate = 3e-4 # PEFT 通常可以用稍大的学习率
   epochs = 3

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # 2. 加载分词器和数据集
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   dataset = load_dataset("glue", dataset_name)

   def preprocess_function(examples):
       # SST-2 数据集包含 'sentence' 和 'label'
       return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

   tokenized_datasets = dataset.map(preprocess_function, batched=True)
   # GLUE 数据集 key 是 'idx', 'sentence', 'label'
   # Trainer 需要 'labels'
   tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
   tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"]) # 移除不需要的列
   tokenized_datasets.set_format("torch") # 设置为 torch 张量格式

   train_dataset = tokenized_datasets["train"]
   eval_dataset = tokenized_datasets["validation"] # GLUE 通常用 validation 集评估

   # 3. 加载基础模型 (不需要加载分类头，PEFT 会处理)
   #    但需要指定 num_labels
   model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

   # 4. 定义 LoRA 配置
   peft_config = LoraConfig(
       task_type=TaskType.SEQ_CLS, # 指定任务类型 (序列分类)
       r=lora_r,
       lora_alpha=lora_alpha,
       target_modules=lora_target_modules, # 指定要应用 LoRA 的层 (通常是 Attention Q/V)
       lora_dropout=lora_dropout,
       bias="none", # 通常不训练 bias
   )

   # 5. 使用 get_peft_model 包装基础模型
   model = get_peft_model(model, peft_config)
   model.print_trainable_parameters() # 打印可训练参数的数量和比例
   # 输出示例: trainable params: 471,810 || all params: 109,950,146 || trainable%: 0.429111...

   # 6. 定义训练参数 (与全参数微调类似，但学习率可能不同)
   training_args = TrainingArguments(
       output_dir=output_dir,
       num_train_epochs=epochs,
       per_device_train_batch_size=batch_size,
       per_device_eval_batch_size=batch_size,
       learning_rate=learning_rate, # 注意学习率可能比 full FT 高
       weight_decay=0.01,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
       metric_for_best_model="accuracy",
       fp16=torch.cuda.is_available(),
       report_to="tensorboard",
   )

   # 7. 定义评估函数
   def compute_metrics(eval_pred):
       logits, labels = eval_pred
       predictions = np.argmax(logits, axis=-1)
       return {"accuracy": accuracy_score(labels, predictions)}

   # 8. 初始化 Trainer
   trainer = Trainer(
       model=model, # 传入的是 PEFT 包装后的模型
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       tokenizer=tokenizer,
       compute_metrics=compute_metrics,
   )

   # 9. 开始训练 (只会训练 LoRA 参数)
   print("Starting LoRA fine-tuning...")
   trainer.train()

   # 10. 评估和保存
   print("\nEvaluating final model...")
   eval_results = trainer.evaluate()
   print(f"Evaluation results: {eval_results}")

   print(f"\nSaving final PEFT adapter model to {output_dir}")
   # 保存 PEFT 适配器权重 (非常小)
   # trainer.save_model(output_dir) # Trainer 会自动保存适配器
   model.save_pretrained(output_dir) # 或者手动调用 save_pretrained
   # 这只会保存 LoRA 的 adapter_config.json 和 adapter_model.bin 文件

   # --- 如何加载和使用 LoRA 微调后的模型 ---
   # from peft import PeftModel, PeftConfig
   # from transformers import AutoModelForSequenceClassification

   # # 1. 加载基础模型
   # base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
   # # 2. 加载 PEFT 配置和权重
   # peft_model = PeftModel.from_pretrained(base_model, output_dir)
   # # 3. 可以直接使用 peft_model 进行推理，或者合并权重
   # merged_model = peft_model.merge_and_unload() # 合并权重，不再需要 PEFT 库

   # # 使用 pipeline
   # from transformers import pipeline
   # pipe = pipeline("text-classification", model=output_dir, device=0 if torch.cuda.is_available() else -1) # 可以直接加载 PEFT 目录
   # result = pipe("This movie feels heartwarming.")
   # print(result)
```

   这个例子展示了使用 `peft` 库应用 LoRA 是多么简单。只需定义 `LoraConfig`，然后用 `get_peft_model` 包装原始模型即可。训练过程与之前类似，但实际更新的只有 LoRA 参数。

**6.3.4 Prefix Tuning / P-Tuning：原理与实现思路**

Prefix Tuning (Li & Liang, 2021) 和 P-Tuning (Liu et al., 2021) 系列方法另辟蹊径，它们不修改模型权重，而是通过在输入层或中间层添加可训练的**连续向量 (Continuous Vectors)** 来引导模型的行为。

* **Prefix Tuning:**
  * **原理:** 为 Transformer 的**每一层**的**键 (Key)** 和**值 (Value)** 向量前，添加一小段**可训练**的**前缀向量 (Prefix Vectors)**。这些前缀向量就像是任务特定的“指令”，在注意力计算时影响着后续 Token 的表示。原始的模型参数保持冻结。
  * **实现:** 需要修改模型的注意力计算部分，将前缀向量拼接到 K 和 V 矩阵的前面。训练时只更新这些前缀向量的参数。
  * **挑战:** 直接优化前缀向量可能不稳定，原论文提出使用一个小的 MLP (重参数化网络) 来生成前缀向量以提高稳定性。
* **P-Tuning:**
  * **P-Tuning v1:** 只在**输入嵌入层**引入少量可训练的**伪标记 (Pseudo Tokens)** 或 **虚拟标记 (Virtual Tokens)**，它们的嵌入向量是可学习的。这些虚拟标记的作用类似于离散的 Prompt，但其表示是连续优化的。
  * **P-Tuning v2 (Deep Prompt Tuning):** (Liu et al., 2022) 发现 v1 在复杂 NLU 任务上效果有限且不稳定。v2 借鉴了 Prefix Tuning 的思想，在模型的**每一层**都添加了可训练的 Prompt 向量（类似于 Prefix），而不仅仅是在输入层。这使得 Prompt 能够更深层次地影响模型的内部表示。P-Tuning v2 在多种 NLU 任务上取得了与全参数微调相当甚至更好的性能，且参数效率很高。
* **优点:** 参数效率高，不改变原始模型权重。
* **缺点:** Prefix Tuning 和 P-Tuning v2 需要修改模型内部结构；P-Tuning v1 性能相对有限。推理时需要额外处理这些前缀/Prompt向量。

**6.3.5 Prompt Tuning：原理与实现思路**

Prompt Tuning (Lester et al., 2021) 是 P-Tuning v1 的一种简化和变体，也是参数效率最高的方法之一。

* **原理:** 只在模型的**输入嵌入层**，将一小段**可训练**的**连续提示向量 (Soft Prompt / Continuous Prompt)** 添加到原始输入的 Token 嵌入序列**之前**。这些软提示向量的长度通常很短（如 10-100 个向量）。在训练时，模型的**所有原始参数完全冻结**，**只更新**这些软提示向量。
* **实现:** 非常简单，只需修改输入嵌入的获取方式，将可学习的软提示嵌入拼接到序列开头。
* **优点:**
  * 参数效率**极高**（只训练几千到几万个参数）。
  * 实现非常简单，不修改模型架构。
  * 易于存储和切换任务（每个任务只需存储其对应的软提示向量）。
* **缺点:**
  * 性能通常**略低于** LoRA、Adapter 或全参数微调，尤其是在较小的模型或较复杂的任务上。对于非常大的模型（>10B 参数），Prompt Tuning 的性能可以接近全参数微调。
  * 对软提示的长度和初始化比较敏感。

**6.3.6 PEFT方法比较与选择**

| 方法                    | 主要思想               | 修改位置              | 可训练参数              | 性能 (相对 Full FT) | 推理延迟 (不合并) | 推理延迟 (合并)  |
| :---------------------- | :--------------------- | :-------------------- | :---------------------- | :------------------ | :---------------- | :--------------- |
| **Full FT**       | 训练所有参数           | 所有层                | 100%                    | Baseline            | Baseline          | Baseline         |
| **Adapter**       | 插入小瓶颈层           | 每层内部 (Attn/FFN后) | 低 (0.1-1%)             | 接近                | 增加              | 不可合并         |
| **LoRA**          | 低秩更新矩阵           | 特定层权重 (Attn Q/V) | 低 (0.1-1%)             | 接近 / 可能超过     | 略微增加          | **无增加** |
| **Prefix Tuning** | 添加 K/V 前缀向量      | 每层 Attention        | 低 (<0.1%)              | 接近                | 增加              | 不可合并         |
| **P-Tuning v2**   | 添加每层 Prompt 向量   | 每层                  | 低 (<0.1%)              | 接近 / 可能超过     | 增加              | 不可合并         |
| **Prompt Tuning** | 添加输入层 Soft Prompt | 输入嵌入层            | **极低** (<0.01%) | 稍低 / 大模型接近   | 略微增加          | 不可合并         |

**如何选择？**

* **追求最佳性能，资源充足:** 全参数微调仍然是黄金标准，尤其是在数据充足的情况下。
* **性能接近 Full FT，且推理延迟重要:** **LoRA** 是一个非常好的选择，因为它可以在合并权重后不增加推理开销。
* **性能要求高，可接受少量推理延迟:** Adapter 或 P-Tuning v2 也是不错的选择。
* **极度关注参数效率和存储，可接受性能略微下降:** **Prompt Tuning** 是最经济的选择，尤其是在模型规模非常大时。
* **易用性:** LoRA 和 Prompt Tuning (尤其是借助 `peft` 库) 通常实现起来最简单。

实践中，LoRA 因其良好的性能、无需增加推理延迟（合并后）以及易用性，成为了目前最受欢迎的 PEFT 方法之一。

**6.4 指令微调（Instruction Tuning）**

前面讨论的微调方法（无论是 Full FT 还是 PEFT）通常是针对**单一特定任务**进行的。模型在一个任务上微调后，虽然在该任务上表现出色，但可能无法很好地泛化到其他**未见过**的任务。

**指令微调 (Instruction Tuning)** 是一种特殊的微调范式，其目标是提升模型**遵循自然语言指令**的能力，并增强其在**多种任务上的零样本泛化能力**。

**6.4.1 目标：提升模型遵循指令的能力和泛化性**

核心思想是：不让模型只学习解决某个特定任务（如“情感分类”），而是让它学习一个更通用的能力——“**根据指令完成任务**”。通过在一个**包含大量不同任务指令的数据集**上进行微调，模型可以学会理解指令的意图，并调用其预训练时学到的知识和能力来执行这些指令。

**好处:**

* **增强零样本/少样本能力:** 模型在指令微调后，即使遇到训练时未明确见过的、但可以用自然语言指令描述的新任务，也能展现出不错的解决能力。
* **提升交互性:** 使模型更像一个“助手”，能够理解并执行用户的各种指令。
* **统一多种任务:** 可以将各种 NLP 任务（分类、生成、问答、摘要、翻译、推理等）统一到“指令执行”这一个框架下。

**6.4.2 指令数据集的构建（FLAN, Alpaca等）**

指令微调的关键在于构建一个**大规模、高质量、多样化**的指令数据集。数据集通常包含成千上万个 **(指令, 输入 (可选), 输出)** 的三元组实例。

* **指令 (Instruction):** 对任务的自然语言描述。例如：“判断以下句子的情感是积极还是消极。”，“将下面的文章总结成三句话。”，“回答以下问题：法国的首都是哪里？”。
* **输入 (Input):** 任务所需的额外上下文信息（可选）。例如，情感分类任务中的句子，摘要任务中的文章，问答任务中的问题和背景段落。
* **输出 (Output):** 模型应该生成的、符合指令要求的答案或结果。

**构建方法:**

1. **基于现有 NLP 数据集改造:** 将各种公共的 NLP 数据集（如 GLUE, SuperGLUE, SQuAD, CNN/DailyMail 等）转换成指令格式。研究者会为每个原始数据集设计多种不同的指令模板，以增加多样性。Google 的 **FLAN (Fine-tuned Language Net)** (Wei et al., 2021) 和 **T0 (Training Transformers to Transform Text)** (Sanh et al., 2021) 就是采用这种方式构建了包含数百个任务和数千个指令模板的数据集。
2. **人工编写:** 由人类专家或众包人员针对各种可能的任务编写指令和对应的输入输出。成本高，但可能质量更高、更有创意。
3. **基于强大模型生成 (Self-Instruct / Alpaca):** 利用一个非常强大的“教师”模型（如 GPT-3.5 或 GPT-4）来自动生成指令、输入和输出。
   * **Self-Instruct (Wang et al., 2022):** 提供少量种子指令，让教师模型生成更多样化的指令，然后让教师模型基于这些指令生成输入和输出实例，并进行过滤。
   * **Alpaca (Taori et al., 2023):** 采用 Self-Instruct 方法，使用 OpenAI 的 `text-davinci-003` 模型生成了约 5 万条指令数据，并用这些数据微调了 Llama 模型，得到了 Alpaca 模型，展现出惊人的指令遵循能力。后续很多开源指令微调项目都借鉴了这种方法。

**数据集的多样性至关重要，需要覆盖:**

* **任务多样性:** 分类、生成、提取、改写、问答、推理等。
* **指令措辞多样性:** 同一个任务可以用不同的方式提问。
* **输入/输出格式多样性:** 不同的任务有不同的输入输出形式。
* **领域多样性:** 覆盖常识、科学、历史、文化等多个领域。

**6.4.3 指令微调的流程与技巧**

指令微调的流程与标准的监督微调（Supervised Fine-tuning, SFT）类似：

1. **选择基础模型:** 通常选择一个强大的预训练语言模型（Decoder-Only 模型如 Llama, GPT-NeoX, Falcon 等是常见的选择）。
2. **准备指令数据集:** 获取或构建如上所述的大规模指令数据集。
3. **格式化数据:** 将每个 (指令, 输入, 输出) 实例格式化成模型期望的单一输入序列。通常会使用特定的模板，例如：
   ```
   下面是一个描述任务的指令，以及一个提供进一步上下文的输入。请写一个恰当的响应来完成请求。

   ### 指令:
   {instruction}

   ### 输入:
   {input}

   ### 响应:
   {output}
   ```

   或者更简洁的模板。将整个格式化后的文本作为模型的输入（或部分作为输入，响应部分作为标签）。
4. **进行微调:** 使用标准的监督学习目标（通常是 CLM 损失，只计算“响应”部分的损失）在指令数据集上微调模型。可以使用全参数微调或 PEFT 方法（如 LoRA）。
5. **评估:** 评估指令微调后的模型通常比较复杂，因为目标是提升通用指令遵循能力。常用的方法包括：
   * 在**未见过的任务**（指令格式可能也不同）上进行零样本评估。
   * 使用特定的**指令遵循能力基准测试**（如 BIG-bench 中的部分任务，HELM 中的评估）。
   * **人工评估:** 由人类判断模型生成的响应是否准确、相关、符合指令要求。

**技巧:**

* **数据质量和多样性是关键:** 比起单纯增加数据量，保证指令的多样性和输出的质量更重要。
* **模板选择:** 数据格式化的模板会影响模型学习效果。
* **训练策略:** 超参数（学习率、批次大小等）需要调整。对于长序列，可能需要梯度检查点等技术。

**6.4.4 案例：展示指令微调前后模型在未见过任务上的表现差异（概念性）**

假设我们有一个基础的预训练语言模型 (Base LLM) 和一个经过指令微调的模型 (Instruction-Tuned LLM)。我们给它们一个它们在训练阶段（包括预训练和指令微调数据）都**没有明确见过**的任务指令：

**指令:** "将以下句子改写成被动语态： 'The cat chased the mouse.'"

* **Base LLM (未指令微调):** 可能无法理解“被动语态”这个指令，或者忽略它，可能输出：
  * "The cat chased the mouse quickly." (只是续写)
  * "The mouse was running from the cat." (语义相关但不符指令)
  * "The cat and the mouse..." (无法执行任务)
* **Instruction-Tuned LLM:** 由于它在训练中见过各种改写、翻译、遵循语法规则的指令，它更有可能理解“被动语态”的含义，并正确执行：
  * "The mouse was chased by the cat." (正确输出)

这个简单的例子说明了指令微调如何赋予模型更好的理解和执行未见过指令的能力，从而提高其泛化性和实用性。许多表现惊艳的聊天机器人（如 ChatGPT 的早期版本、Alpaca、Vicuna 等）都受益于大规模的指令微调。

**6.5 多任务微调与持续学习**

除了上述主要的微调策略，还有一些相关的概念：

* **多任务微调 (Multi-Task Fine-tuning):** 同时在**多个相关任务**的数据集上对模型进行微调。目标是让模型学习到任务之间的共性，相互促进，提升在所有这些任务上的表现，并可能提高泛化能力。实现方式可以是混合来自不同任务的数据批次，或者为不同任务设计不同的任务头但共享底层模型参数。
* **持续学习 (Continual Learning / Lifelong Learning):** 让模型能够**按顺序学习新任务**，同时**不忘记**之前学过的任务（克服灾难性遗忘）。这是机器学习中的一个长期挑战。对于大模型，如何在不重新训练整个模型的情况下，持续地为其注入新知识或适配新任务，是一个重要的研究方向。PEFT 方法（如 Adapter, LoRA）由于其模块化和参数隔离的特性，被认为在持续学习场景下具有潜力。

**6.6 本章小结：微调策略的选择**

本章我们详细探讨了如何通过**微调**将预训练大模型的通用能力引导至特定任务或提升其通用指令遵循能力。

* **为何微调:** 当零样本/少样本提示无法满足特定任务的性能、稳定性或深度知识要求时，微调是必要的步骤。
* **全参数微调 (Full FT):** 更新所有模型参数，通常性能最好，但存储和计算成本高。适用于资源充足且追求极致性能的场景。
* **参数高效微调 (PEFT):** 只训练模型参数的一小部分，显著降低成本。
  * **Adapter Tuning:** 插入小型瓶颈层。
  * **LoRA:** 学习权重的低秩更新，推理时可合并，不增加延迟，是当前非常流行的方法。
  * **Prefix/P-Tuning/Prompt Tuning:** 在输入或中间层添加可训练的连续向量。Prompt Tuning 参数效率最高。
  * 选择 PEFT 方法需权衡性能、参数效率、推理开销和实现复杂度。
* **指令微调 (Instruction Tuning):** 通过在大量多样化的 (指令, 输入, 输出) 数据上微调，提升模型遵循自然语言指令和零样本泛化到未见任务的能力。是构建通用助手型大模型的关键技术。

微调是释放大模型潜能、使其在实际应用中落地的关键环节。选择哪种微调策略取决于具体的目标任务、性能要求、可用数据、计算资源预算以及部署约束。

在模型经过预训练和（可能的）微调之后，我们还需要解决另一个至关重要的问题：如何确保模型的行为不仅强大，而且符合人类的价值观和期望？这就是下一章——**第7章：人类对齐：让大模型更符合人类期望**——将要探讨的核心内容。

---
