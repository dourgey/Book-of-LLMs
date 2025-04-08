**第一部分：基础与背景**

# 第2章：必备基础知识回顾

在深入探索大模型的奥秘之前，我们需要确保已经牢固掌握了构建和理解它们所必需的基础知识。大模型并非空中楼阁，它深深植根于深度学习的核心原理、自然语言处理的基本技术以及高效的编程实践。本章旨在快速回顾这些关键的先决条件，重点梳理那些与后续章节内容紧密相关的概念和工具。

我们将重温深度学习的基石，如神经网络、反向传播和优化算法；梳理自然语言处理中的核心环节，特别是文本表示和至关重要的分词技术；最后，确保你已准备好Python与PyTorch的开发环境，以便顺利运行本书中的代码示例。即使你对这些领域已有了解，温故知新也能帮助你更好地将这些基础知识与大模型的特定挑战和解决方案联系起来。

**2.1 深度学习核心概念复习**

深度学习是驱动大模型发展的核心引擎。以下是一些关键概念的回顾。

**2.1.1 神经网络与反向传播**

* **神经网络 (Neural Network):** 受到生物神经系统启发，神经网络是由大量相互连接的节点（神经元）组成的计算模型。最简单的形式是前馈神经网络（Feedforward Neural Network），信息从输入层单向流经一个或多个隐藏层（Hidden Layers），最终到达输出层。每个连接都有一个权重（Weight），每个神经元（除输入层外）通常有一个偏置（Bias）和一个激活函数（Activation Function）。

  * **深度神经网络 (Deep Neural Network, DNN):** 指拥有多个隐藏层的神经网络。层数的增加使得网络能够学习更复杂、更抽象的数据表示。
  * **计算过程（前向传播 Forward Propagation）:** 给定输入 `x`，网络通过一系列线性变换（加权求和加偏置）和非线性激活函数，逐层计算，最终得到输出 `ŷ`。例如，一个简单的隐藏层计算可以表示为 `h = σ(Wx + b)`，其中 `W` 是权重矩阵，`b` 是偏置向量，`σ` 是激活函数。
* **反向传播 (Backpropagation):** 这是训练神经网络的核心算法。其目标是根据模型预测输出 `ŷ` 与真实标签 `y` 之间的误差（通过损失函数衡量），计算损失函数相对于网络中每个参数（权重和偏置）的梯度（Gradient），然后利用这些梯度通过优化算法来更新参数，以最小化损失。

  * **链式法则 (Chain Rule):** 反向传播本质上是应用微积分中的链式法则，从输出层开始，逐层向后计算梯度。它高效地计算了损失对网络中所有参数的偏导数。
  * **梯度 (Gradient):** 梯度是一个向量，指向函数值增长最快的方向。在优化中，我们通常沿着负梯度方向更新参数，以期望找到损失函数的最小值。`∇L = (∂L/∂w₁, ∂L/∂w₂, ...)`。

**2.1.2 损失函数与优化器**

* **损失函数 (Loss Function / Cost Function / Objective Function):** 用于衡量模型预测值 `ŷ` 与真实值 `y` 之间差距的函数。训练的目标就是最小化这个函数的值。常见的损失函数包括：

  * **均方误差 (Mean Squared Error, MSE):** 常用于回归任务。`L = (1/N) * Σ(yᵢ - ŷᵢ)²`。
  * **交叉熵损失 (Cross-Entropy Loss):** 常用于分类任务，特别是多分类问题。对于语言模型（预测下一个词的概率分布），交叉熵损失是标准的衡量指标。它衡量了模型预测的概率分布与真实的（通常是One-hot）概率分布之间的差异。
    * *对数似然角度理解：* 最小化交叉熵等价于最大化观测数据的对数似然。对于预测下一个词的任务，目标是最大化真实下一个词的预测概率。
    * *信息论角度理解：* 交叉熵衡量了使用模型预测的分布 `q` 来编码真实分布 `p` 所需要的平均比特数与使用真实分布 `p` 自身来编码所需的最优平均比特数（即熵 `H(p)`）之间的差值 `D_KL(p||q) + H(p)`。当 `p` 是固定（真实标签）时，最小化交叉熵等价于最小化KL散度 `D_KL(p||q)`，即让预测分布 `q` 尽可能接近真实分布 `p`。
* **优化器 (Optimizer):** 根据损失函数计算出的梯度来更新模型参数的算法。目标是找到使损失函数最小化的参数值。

  * **随机梯度下降 (Stochastic Gradient Descent, SGD):** 最基本的优化算法。每次更新使用一小批（mini-batch）数据计算梯度，并沿着负梯度方向更新参数：`θ = θ - η * ∇L(θ; x_batch, y_batch)`，其中 `η` 是学习率（Learning Rate）。
    * *挑战：* 可能陷入局部最优，对学习率敏感，更新方向可能不稳定。
  * **带动量的SGD (SGD with Momentum):** 引入动量项，模拟物理中物体运动的惯性。它累积了过去梯度的指数衰减平均值，有助于加速收敛，越过平坦区域，并抑制震荡：`v = βv + η * ∇L(θ)`, `θ = θ - v`。
  * **Adam (Adaptive Moment Estimation):** (Kingma & Ba, 2014) 目前最常用、效果通常也较好的优化器之一，特别是在训练大型神经网络时。Adam 结合了 Momentum 和 RMSprop 的思想：
    * 它为每个参数计算自适应的学习率。
    * 它同时跟踪梯度的一阶矩（均值，类似Momentum）和二阶矩（未中心的方差）的指数衰减移动平均。
    * 更新规则大致为：`m = β₁m + (1-β₁)∇L`, `v = β₂v + (1-β₂)(∇L)²`, `m_hat = m / (1-β₁^t)`, `v_hat = v / (1-β₂^t)`, `θ = θ - η * m_hat / (sqrt(v_hat) + ε)`。 (其中 `t` 是时间步，`ε` 是为了防止除零的小常数)。
    * *优点：* 通常收敛速度快，对初始学习率相对不敏感（但并非完全不敏感），适用于处理稀疏梯度和非平稳目标。
    * *大模型训练中的常用选择：* AdamW 是 Adam 的一个变种，修复了原始 Adam 中 L2 正则化与权重衰减不等价的问题，在大模型训练中更为常用。

**2.1.3 激活函数**

* **激活函数 (Activation Function):** 引入非线性是神经网络能够学习复杂模式的关键。激活函数作用于神经元的加权输入（`Wx + b`），产生该神经元的输出。如果缺少非线性激活函数，多层神经网络本质上等价于单层线性模型。
  * **Sigmoid:** 将输入压缩到 (0, 1) 之间。`σ(x) = 1 / (1 + exp(-x))`。曾用于早期网络，但易导致梯度消失，输出非零中心。
  * **Tanh (双曲正切):** 将输入压缩到 (-1, 1) 之间。`tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`。输出是零中心的，通常比 Sigmoid 效果好，但仍存在梯度消失问题。
  * **ReLU (Rectified Linear Unit):** (Nair & Hinton, 2010) 目前最常用的激活函数之一。`ReLU(x) = max(0, x)`。
    * *优点：* 计算简单高效，在正区间（x>0）梯度恒为1，缓解了梯度消失问题，加速收敛。
    * *缺点：* 输出非零中心；存在“死亡ReLU”问题（Dead ReLU Problem），即如果神经元输出恒为负，其梯度将永远为零，参数无法更新。
  * **Leaky ReLU / PReLU / ELU:** ReLU 的变种，旨在解决死亡ReLU问题，允许负输入有小的非零梯度。
  * **GeLU (Gaussian Error Linear Unit):** (Hendrycks & Gimpel, 2016) 在 Transformer 类模型（如 BERT, GPT-2/3）中广泛使用。`GeLU(x) = x * Φ(x)`，其中 `Φ(x)` 是标准正态分布的累积分布函数。它是一个平滑的、非单调的 ReLU 近似。经验表明在 Transformer 中效果优于 ReLU。
  * **Swish / SiLU (Sigmoid Linear Unit):** (Ramachandran et al., 2017; Elfwing et al., 2018) `Swish(x) = x * σ(βx)`，其中 `β` 是可学习参数或固定为1（此时也叫 SiLU）。与 GeLU 类似，也是平滑的非单调函数，在一些任务上表现优于 ReLU 和 GeLU。

**2.1.4 正则化**

* **正则化 (Regularization):** 用于防止模型过拟合（Overfitting）的技术。过拟合指模型在训练数据上表现很好，但在未见过的测试数据上表现差，即泛化能力差。大型模型由于参数众多，特别容易过拟合。
  * **L1/L2 正则化 (Weight Decay):** 在损失函数中加入参数的 L1 或 L2 范数惩罚项，倾向于使参数值变小，从而简化模型。L2 正则化（权重衰减）更常用。
  * **Dropout:** (Srivastava et al., 2014) 训练过程中，以一定概率 `p` 随机将一部分神经元的输出设置为零。
    * *效果：* 强迫网络学习冗余表示，减少神经元之间的复杂共适应关系，提高模型的鲁棒性和泛化能力。可以看作是同时训练多个共享权重的“瘦”网络，并在测试时进行模型平均的近似。
    * *实现：* 在训练时应用 Dropout，测试时通常关闭 Dropout，并将所有权重乘以 `(1-p)` 或在训练时进行 Inverted Dropout（将未被置零的神经元输出除以 `1-p`）。
  * **Layer Normalization (LayerNorm):** (Ba et al., 2016) Transformer 架构中广泛使用的归一化技术。与主要用于CNN的 BatchNorm 不同，LayerNorm 在**单个样本**的**层内**进行归一化，而不是跨批次。
    * *计算方式：* 对同一层内所有神经元的激活值（在一个样本内）计算均值和方差，然后进行归一化，最后通过可学习的缩放因子（gain, `γ`）和平移因子（bias, `β`）进行仿射变换。`y = (x - μ) / sqrt(σ² + ε) * γ + β`，其中 `μ` 和 `σ²` 是对 `x` 在特征维度上计算的均值和方差。
    * *优点：*
      * 不受批次大小（Batch Size）的影响，在小批量或序列长度可变的RNN/Transformer中表现稳定。
      * 有助于稳定训练过程，平滑损失曲面，允许使用更高的学习率。
      * 是 Transformer 中 Add & Norm 结构的关键组成部分（与残差连接一起）。
  * **Batch Normalization (BatchNorm):** (Ioffe & Szegedy, 2015) 主要在CNN中使用。它在**批次维度**上对每个特征通道进行归一化。计算批次内样本在某个特征上的均值和方差来进行归一化。依赖于批次大小，且需要维护运行时的均值和方差用于推理。在RNN和Transformer中效果通常不如LayerNorm。

理解这些深度学习基础，特别是损失函数（交叉熵）、优化器（AdamW）、激活函数（GeLU/Swish）和正则化（Dropout, LayerNorm），对于理解 Transformer 的内部工作原理和大型模型的训练至关重要。

**2.2 自然语言处理（NLP）基础**

自然语言处理（Natural Language Processing, NLP）是人工智能的一个分支，专注于使计算机能够理解、处理和生成人类语言。大语言模型本质上是极其强大的NLP模型。

**2.2.1 文本表示：从One-hot到分布式表示**

如何将非结构化的文本数据转换为计算机能够处理的数值形式，是NLP的首要问题。

* **One-hot 编码:** 最简单的方法。创建一个包含语料库中所有唯一词的词汇表（Vocabulary）。每个词被表示为一个长度等于词汇表大小的向量，该词对应的索引位置为1，其余位置为0。
  * *缺点：* 向量维度巨大且稀疏（维度等于词汇表大小，通常数万到数百万）；无法表示词语之间的语义相似性（任意两个词的向量点积为0）。
* **词袋模型 (BoW) / TF-IDF:** 如第1章所述，它们将文本表示为词频或TF-IDF值的向量，忽略了词序。同样存在稀疏性问题，且语义表达能力有限。
* **分布式表示 (Distributed Representation) / 词嵌入 (Word Embedding):** （回顾1.2.2节）将每个词映射到一个低维（如几百维）的稠密向量（Dense Vector）。
  * *核心思想：* 词的含义由其上下文决定（Distributional Hypothesis）。通过在大量文本上训练模型（如Word2Vec, GloVe, 或现代深度模型中的嵌入层），使得经常出现在相似上下文中的词具有相似的向量表示。
  * *优点：* 维度相对较低且稠密；能够捕捉词语之间的语义和句法关系（向量空间中的距离和方向有意义）；是现代NLP模型（包括大模型）的标准输入表示。
  * *在大模型中：* 大模型通常包含一个巨大的嵌入层（Embedding Layer），作为模型的第一层。这个层的作用是将输入的词（或更准确地说，是Token，见下文）ID 映射为其对应的分布式向量表示。这些嵌入向量在模型的预训练过程中与其他参数一起被学习。

**2.2.2 分词（Tokenization）：BPE, WordPiece, SentencePiece**

在将文本送入模型之前，我们需要将连续的字符序列切分成一系列有意义的单元，称为**词元（Token）**。这个过程就是**分词（Tokenization）**。选择合适的分词策略对模型性能至关重要。

* **为何需要分词？**

  * **处理词汇表外 (Out-of-Vocabulary, OOV) 问题：** 如果直接按空格切分单词，遇到训练时未见过的词（如新词、拼写错误、专有名词）怎么办？模型将无法处理。
  * **控制词汇表大小：** 如果每个单词都作为一个Token，词汇表会非常庞大，导致嵌入层参数量巨大，计算效率低下。
  * **捕捉构词法 (Morphology):** 像 "running", "ran", "runner" 这些词有共同的词根 "run"。理想的分词策略应能反映这种构词关系。
  * **处理多语言和特殊字符：** 需要一种能鲁棒处理各种语言（包括没有明确空格分隔的语言）和特殊符号的方法。
* **简单分词（如按空格/标点）：** 易于实现，但在处理OOV、构词法和某些语言时效果不佳。
* **子词分词 (Subword Tokenization):** 现代大模型普遍采用的核心策略。它介于字符级分词和词级分词之间。基本思想是将常见词保留为完整单元，同时将稀有词拆分成更小的、有意义的子词单元（Subword Units），甚至单个字符。这样可以：

  * **有效处理OOV：** 任何未登录词都可以由已知的子词组合而成（最坏情况退化为字符）。
  * **显著缩小词汇表：** 通常几万个子词单元就能覆盖大部分文本。
  * **共享统计信息：** 像 "running" 和 "runner" 可能被拆分为 `["run", "##ning"]` 和 `["run", "##ner"]`（`##` 表示该子词是词的一部分，而非词首），它们共享了 "run" 的表示，有助于模型学习构词规律。
* **主流子词分词算法：**

  * **字节对编码 (Byte-Pair Encoding, BPE):** (Sennrich et al., 2015)

    1. **初始化：** 词汇表初始只包含所有单个字符。
    2. **迭代合并：** 统计语料库中相邻Token对的出现频率，将频率最高的Token对合并成一个新的Token，并加入词汇表。
    3. **重复：** 重复步骤2，直到达到预设的词汇表大小（Merge次数）或没有可合并的对。

    * *特点：* 基于频率的贪心算法。GPT系列模型常用。
  * **WordPiece:** (Schuster & Nakajima, 2012; Used by BERT)

    1. **初始化：** 词汇表包含所有单个字符。
    2. **迭代合并：** 选择能够最大化**训练数据似然度（Likelihood）** 增加的相邻Token对进行合并。即将两个单元合并后，如果使得语料库的整体概率（假设子词独立出现）最大，则合并它们。

    * *特点：* 基于似然度的优化。BERT、DistilBERT等模型常用。通常在词内部的子词前加 `##` 标记。
  * **SentencePiece:** (Kudo & Richardson, 2018)

    1. **视空格为普通字符：** 将空格也视为一种特殊字符（如 `_`）并包含在子词切分中。这使得分词和解码过程完全可逆，且不依赖于特定语言的空格规则。
    2. **基于Unigram LM或BPE/WordPiece：** 可以使用不同的算法构建词汇表和进行分词，其中 Unigram Language Model 算法试图从一个较大的候选子词集合中移除一些子词，使得移除后语料库的似然度下降最少，最终保留一个最优的子词词汇表。

    * *特点：* 语言无关，直接在原始文本流上操作，解码方便。T5、XLNet、Llama等模型常用。
* **特殊词元 (Special Tokens):** 在分词过程中，通常会加入一些具有特殊含义的Token，用于模型的输入格式化：

  * `[CLS]` (Classification): 通常放在序列开头，其对应的输出向量可用于句子级别的分类任务（如BERT）。
  * `[SEP]` (Separator): 用于分隔两个句子或文本片段（如BERT的NSP任务输入）。
  * `[PAD]` (Padding): 用于将同一批次中不同长度的序列填充到相同长度。
  * `[UNK]` (Unknown): 代表词汇表中不存在的Token（虽然子词分词大大减少了UNK的出现）。
  * `[MASK]` (Masking): 在MLM预训练任务中用于替换被遮盖的Token。
  * `<s>`, `</s>`: 分别表示序列开始和结束（常用在生成模型中）。
  * `<|endoftext|>`: GPT系列常用的序列结束符。

**案例：使用Hugging Face `tokenizers` 库演示BPE分词**

Hugging Face 的 `tokenizers` 库提供了高效的、纯Python绑定的Rust实现的主流分词算法。下面是一个简单的BPE分词器训练和使用的例子：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace # 或者更复杂的预分词器
from tokenizers.processors import TemplateProcessing # 用于添加特殊Token

# 1. 初始化 BPE 模型和 Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 设置预分词器 (Pre-tokenizer) - 如何初步切分文本
#    这里使用简单的按空格切分，实际应用可能更复杂
tokenizer.pre_tokenizer = Whitespace()

# 3. 准备训练数据 (一些示例文本文件路径)
#    实际应用中需要大量文本数据
corpus_files = ["path/to/your/text_file1.txt", "path/to/your/text_file2.txt"]
# 假设文件内容是纯文本，每行一个句子或段落

# 4. 设置训练器 (Trainer) - 定义 BPE 训练参数
#    vocab_size: 最终词汇表大小
#    special_tokens: 训练时需要考虑的特殊Token
#    min_frequency: 一个词元对至少出现多少次才会被合并
trainer = BpeTrainer(vocab_size=10000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], min_frequency=2)

# 5. 训练 Tokenizer
print("Training tokenizer...")
tokenizer.train(files=corpus_files, trainer=trainer)
print("Training complete.")

# 6. 保存训练好的 Tokenizer
tokenizer.save("my_bpe_tokenizer.json")

# 7. 加载并使用 Tokenizer
print("\nLoading tokenizer...")
loaded_tokenizer = Tokenizer.from_file("my_bpe_tokenizer.json")

# -- (可选) 添加后处理器 (Post-processor) 来自动添加特殊 Token --
# 例如，对于 BERT 风格的输入 "[CLS] sentence [SEP]"
loaded_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1", # 用于处理句子对输入
    special_tokens=[
        ("[CLS]", loaded_tokenizer.token_to_id("[CLS]")),
        ("[SEP]", loaded_tokenizer.token_to_id("[SEP]")),
    ],
)

# 8. 对新文本进行编码 (Encoding)
text = "Let's learn tokenization with Hugging Face tokenizers!"
print(f"\nOriginal text: {text}")
encoding = loaded_tokenizer.encode(text)

print("Encoded output:")
print(f"  Tokens: {encoding.tokens}") # 查看切分出的 Token 字符串
print(f"  IDs: {encoding.ids}")       # 查看对应的 Token ID (送入模型的输入)
print(f"  Attention Mask: {encoding.attention_mask}") # 标记哪些是真实Token，哪些是Padding
# 注意：因为我们加了 post_processor, 输出会自动包含 [CLS] 和 [SEP]

# 9. 进行解码 (Decoding) - 将 ID 转换回文本
decoded_text = loaded_tokenizer.decode(encoding.ids, skip_special_tokens=True) # 跳过特殊Token
print(f"\nDecoded text (skip special tokens): {decoded_text}")

decoded_text_with_special = loaded_tokenizer.decode(encoding.ids, skip_special_tokens=False)
print(f"Decoded text (with special tokens): {decoded_text_with_special}")

# 示例输出 (会根据你的训练数据和参数变化):
# Original text: Let's learn tokenization with Hugging Face tokenizers!
# Encoded output:
#   Tokens: ['[CLS]', 'Let', "'", 's', 'learn', 'token', 'ization', 'with', 'Hu', 'gging', 'Face', 'token', 'izers', '!', '[SEP]']
#   IDs: [1, 89, 15, 8, 223, 567, 1234, 98, 2500, 3500, 1800, 567, 6789, 30, 2]
#   Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#
# Decoded text (skip special tokens): Let's learn tokenization with Hugging Face tokenizers!
# Decoded text (with special tokens): [CLS] Let's learn tokenization with Hugging Face tokenizers! [SEP]
```

这个例子展示了使用 `tokenizers` 库进行BPE分词的基本流程：初始化、设置预分词、训练、保存/加载、编码和解码。理解这个过程对于后续使用预训练模型（它们都自带了分词器）或进行模型预训练至关重要。

**2.2.3 词汇表（Vocabulary）构建**

分词过程的产物之一就是**词汇表 (Vocabulary)**。它是一个映射，将每个唯一的词元（Token）映射到一个整数ID。模型实际处理的是这些ID序列，而不是原始文本。词汇表通常包含：

* 所有通过分词算法产生的子词单元。
* 所有定义的特殊词元（`[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]` 等）。

词汇表的大小（`vocab_size`）是一个重要的超参数，需要在模型性能、内存占用和计算效率之间进行权衡。现代大模型的词汇表大小通常在 3 万到 10 万以上不等。

**2.2.4 文本预处理流程**

将原始文本转化为模型可接受的输入，通常需要一个预处理流程：

1. **文本清洗 (Cleaning):** 去除HTML标签、特殊噪声字符、处理编码问题等。
2. **文本规范化 (Normalization):** 转换为小写、处理数字、统一标点符号、去除停用词（可选，大模型通常不去除）、词形还原/词干提取（可选，子词分词在一定程度上解决了这个问题）等。
3. **分词 (Tokenization):** 使用选定的分词器（如BPE, WordPiece, SentencePiece）将规范化后的文本切分为词元序列。
4. **转换为ID (Conversion to IDs):** 根据构建好的词汇表，将词元序列映射为整数ID序列。
5. **填充与截断 (Padding & Truncation):** 为了将同一批次（Batch）中的序列处理成相同长度（模型通常需要固定长度的输入），需要对较短的序列进行填充（通常使用 `[PAD]` Token的ID），对较长的序列进行截断。需要确定最大序列长度（`max_length`），这是一个重要的超参数。
6. **生成注意力掩码 (Attention Mask):** 创建一个与ID序列等长的二进制掩码，用于告诉模型哪些Token是真实的输入，哪些是填充的 `[PAD]` Token。真实Token对应的掩码值为1，`[PAD]` Token对应的掩码值为0。自注意力机制在计算时会忽略掉掩码值为0的位置。
7. **(可选) 生成Token类型ID (Token Type IDs / Segment IDs):** 对于需要区分不同文本片段输入的任务（如BERT处理句子对），需要生成一个额外的序列，标记每个Token属于哪个片段（例如，第一个句子为0，第二个句子为1）。

理解这个流程，特别是分词、ID转换、填充和注意力掩码，对于使用和训练大模型至关重要。Hugging Face 的 `transformers` 库极大地简化了这个过程，其提供的 `tokenizer` 对象通常能一站式完成上述3-7步。

**2.3 Python与PyTorch环境准备**

本书的代码示例将主要使用Python语言和PyTorch深度学习框架。确保你的开发环境配置正确是进行实践的第一步。

**2.3.1 Python环境配置（conda/venv）**

为了避免不同项目之间的库版本冲突，强烈建议使用虚拟环境来管理Python包。常用的工具有：

* **Conda:** 一个开源的包管理系统和环境管理系统，可以轻松创建、保存、加载和切换环境。特别适合管理包含非Python依赖（如CUDA工具包）的环境。
  ```bash
  # 创建一个名为 llm_tutorial 的新环境，指定 Python 版本
  conda create -n llm_tutorial python=3.10

  # 激活环境
  conda activate llm_tutorial

  # 在环境中安装包 (示例)
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # 官方推荐安装PyTorch的方式
  pip install transformers datasets tokenizers accelerate # 安装 Hugging Face 生态库

  # 退出环境
  conda deactivate
  ```
* **venv:** Python自带的虚拟环境工具，轻量级。
  ```bash
  # 创建虚拟环境 (在项目目录下)
  python -m venv .venv

  # 激活环境 (Linux/macOS)
  source .venv/bin/activate
  # 激活环境 (Windows)
  # .venv\Scripts\activate

  # 在环境中安装包
  pip install torch torchvision torchaudio
  pip install transformers datasets tokenizers accelerate

  # 退出环境
  deactivate
  ```

选择哪种工具取决于个人偏好和项目需求。对于涉及复杂依赖（特别是GPU驱动和CUDA）的项目，Conda通常更方便。

**核心依赖库：**

* `torch`: PyTorch 核心库。
* `transformers`: Hugging Face 库，提供数千个预训练模型（包括BERT, GPT, T5, Llama等）、分词器和各种NLP工具。
* `datasets`: Hugging Face 库，方便加载和处理各种文本和多模态数据集。
* `tokenizers`: Hugging Face 库，提供高效的分词器实现。
* `accelerate`: Hugging Face 库，简化PyTorch分布式训练和混合精度训练的配置。
* `numpy`: Python 科学计算的基础库。
* (可选) `scikit-learn`: 用于机器学习任务和评估指标。
* (可选) `pandas`: 用于数据处理和分析。
* (可选) `matplotlib`, `seaborn`: 用于数据可视化。

请确保安装与你的硬件（特别是GPU和CUDA版本）兼容的PyTorch版本。

**2.3.2 PyTorch核心API简介**

PyTorch 是一个开源的机器学习框架，以其灵活性、易用性和强大的GPU加速能力而受到研究人员和开发者的青睐。以下是几个核心概念：

* **Tensor:** PyTorch 中最基本的数据结构，类似于 NumPy 的 ndarray，但可以在GPU上进行计算以实现加速。

  ```python
  import torch

  # 创建 Tensor
  x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
  y = torch.rand(2, 2) # 随机初始化

  print("Tensor x:\n", x)
  print("Tensor y:\n", y)

  # Tensor 运算 (可在 CPU 或 GPU 上执行)
  z = x + y
  print("x + y:\n", z)
  print("Matrix multiplication (x @ y):\n", x @ y) # 或者 torch.matmul(x, y)
  ```
* **Autograd (自动微分):** PyTorch 的核心功能之一。如果你创建一个 Tensor 并设置 `requires_grad=True`，PyTorch 会自动追踪在该 Tensor 上的所有操作。当计算完成后，你可以调用 `.backward()` 方法，PyTorch 会自动计算所有 `requires_grad=True` 的 Tensor 相对于某个标量值（通常是损失函数值）的梯度。

  ```python
  # 创建需要梯度的 Tensor
  a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
  b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)

  Q = 3*a**3 - b**2

  # 假设 Q 是某个复杂计算的中间结果，最终得到一个标量损失 L
  L = Q.mean() # 计算均值作为示例损失

  # 反向传播计算梯度
  L.backward()

  # 查看梯度 dL/da 和 dL/db
  print("Gradient of L w.r.t. a (dL/da):\n", a.grad)
  print("Gradient of L w.r.t. b (dL/db):\n", b.grad)
  ```

  `Autograd` 是实现反向传播算法的基础，使得我们无需手动推导复杂的梯度公式。
* **`torch.nn.Module`:** 所有神经网络模块的基类。构建自定义网络或使用预定义层（如线性层、卷积层、RNN层、Transformer层）时，都需要继承 `nn.Module`。

  * 必须实现 `__init__` 方法，用于定义网络的层和参数。
  * 必须实现 `forward` 方法，定义数据在网络中的前向传播路径。

  ```python
  import torch.nn as nn
  import torch.nn.functional as F

  class SimpleNet(nn.Module):
      def __init__(self, input_size, hidden_size, output_size):
          super(SimpleNet, self).__init__() # 必须调用父类的 __init__
          self.fc1 = nn.Linear(input_size, hidden_size) # 定义第一个全连接层
          self.relu = nn.ReLU() # 定义 ReLU 激活
          self.fc2 = nn.Linear(hidden_size, output_size) # 定义第二个全连接层

      def forward(self, x):
          # 定义前向传播逻辑
          out = self.fc1(x)
          out = self.relu(out)
          out = self.fc2(out)
          # 注意：这里通常不直接加 Softmax，因为 nn.CrossEntropyLoss 会处理
          return out

  # 使用网络
  input_dim = 784 # 例如 MNIST 图片展平
  hidden_dim = 128
  output_dim = 10   # 10 个类别
  model = SimpleNet(input_dim, hidden_dim, output_dim)
  print(model)

  # 随机输入
  dummy_input = torch.randn(64, input_dim) # batch_size=64
  output = model(dummy_input)
  print("Output shape:", output.shape) # torch.Size([64, 10])
  ```
* **`torch.optim`:** 包含各种优化算法的实现（如 `optim.SGD`, `optim.Adam`, `optim.AdamW`）。

  ```python
  import torch.optim as optim

  # 定义优化器，传入模型参数和学习率
  optimizer = optim.AdamW(model.parameters(), lr=0.001)

  # 在训练循环中：
  # optimizer.zero_grad() # 清除上一轮的梯度
  # loss = compute_loss(output, target) # 计算损失
  # loss.backward()       # 反向传播计算梯度
  # optimizer.step()        # 更新模型参数
  ```
* **`torch.nn.functional` (F):** 提供了许多常用的函数式接口，如激活函数 (`F.relu`, `F.gelu`), 损失函数 (`F.cross_entropy`), 池化操作等。它们与 `nn.Module` 形式的层的区别在于，函数式接口是无状态的（不包含可学习参数），而 `nn.Module` 可以包含参数和状态。

熟悉这些核心组件将使你能够理解和编写 PyTorch 代码，从而实现本书中的模型和算法。

**2.3.3 GPU加速配置与检查**

训练大模型通常需要 GPU 来加速计算。PyTorch 提供了方便的接口来利用 GPU。

* **检查 GPU 是否可用:**

  ```python
  import torch

  if torch.cuda.is_available():
      print(f"CUDA (GPU) is available!")
      print(f"Number of GPUs: {torch.cuda.device_count()}")
      print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
      device = torch.device("cuda") # 默认使用第一个 GPU (cuda:0)
      # device = torch.device("cuda:1") # 使用第二个 GPU
  else:
      print("CUDA (GPU) is not available, using CPU.")
      device = torch.device("cpu")
  ```
* **将模型和数据移动到 GPU:** 要在 GPU 上进行计算，需要将模型参数和输入数据都显式地移动到指定的 `device`。

  ```python
  # 将模型移动到 GPU (inplace 操作)
  model.to(device)

  # 在训练循环中，将每个批次的数据移动到 GPU
  for inputs, labels in dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      # ... 后续计算 ...
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      # ...
  ```

确保正确配置了 CUDA 环境并安装了对应 CUDA 版本的 PyTorch 是使用 GPU 的前提。

**2.4 本章小结**

本章我们快速回顾了深入学习大模型所必需的基础知识：

* **深度学习核心：** 重温了神经网络的基本结构、训练的关键（反向传播、损失函数、优化器如AdamW）、激活函数（特别是ReLU、GeLU、Swish）以及防止过拟合的正则化技术（Dropout、LayerNorm）。Layer Normalization 对于理解 Transformer 尤为重要。
* **自然语言处理基础：** 强调了从稀疏表示（One-hot）到稠密分布式表示（词嵌入）的转变。详细介绍了现代大模型处理文本的核心技术——**子词分词**（BPE, WordPiece, SentencePiece），并通过 Hugging Face `tokenizers` 库进行了实例演示。理解分词、词汇表、填充、注意力掩码等预处理步骤是使用大模型的前提。
* **Python与PyTorch环境：** 强调了使用虚拟环境（conda/venv）的重要性，并简要介绍了 PyTorch 的核心 API（Tensor, Autograd, nn.Module, optim）以及如何利用 GPU 进行加速计算。

这些基础知识如同一砖一瓦，将支撑起我们后续构建和理解 Transformer 这一宏伟建筑。如果你对本章的任何内容感到生疏，建议在继续阅读前花些时间查阅相关资料或教程进行巩固。

现在，我们已经做好了准备，将在下一章——**第3章：Transformer：大模型的基石**——深入探索那个彻底改变了NLP乃至整个AI领域的革命性架构。

---
