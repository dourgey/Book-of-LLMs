**第六部分：应用与实践**

# 第10章：大模型的典型应用场景

经过前面深入的技术探讨，我们已经对大模型的原理、训练和评估有了扎实的理解。现在，是时候将目光投向这些强大模型在现实世界中的广阔应用了。大语言模型（LLMs）并非仅仅是学术界的宠儿或科技巨头的炫技工具，它们正以前所未有的速度渗透到各行各业，成为驱动创新、提升效率、甚至创造全新体验的核心引擎。

本章将概览大模型的一些最典型、最具影响力的应用场景。我们将看到，无论是需要创造力的文本生成，还是需要严谨性的信息检索，抑或是需要效率的代码编写，大模型都能以其独特的通用性和强大的能力扮演重要角色。通过了解这些应用场景，你不仅能更深刻地体会到学习大模型的价值，也能为思考如何将这些技术应用于你自己的领域或项目中获得启发。我们将结合简单的案例和概念性代码示例（利用 Hugging Face `pipeline` 或其他工具）来直观展示这些应用。

**10.1 文本生成与创作 (Text Generation & Creation)**

这是大语言模型（尤其是 Decoder-Only 架构如 GPT 系列）最核心、最引人注目的能力之一。模型利用其在海量文本数据上学到的语言模式、风格和知识，能够生成各种类型的文本内容。

* **应用方式:**
  * **零样本/少样本提示:** 通过精心设计的 Prompt 直接引导模型生成所需内容（见第8章）。这是最灵活、最常用的方式。
  * **微调:** 在特定风格（如莎士比亚风格，见 6.2.4 案例）、特定主题或特定格式的数据集上微调模型，使其生成更符合要求的文本。
* **典型场景:**
  * **创意写作:** 生成诗歌、歌词、故事、剧本、小说片段等。
  * **内容营销:** 撰写广告文案、社交媒体帖子、博客文章、产品描述。
  * **邮件与报告撰写:** 起草邮件、生成会议纪要、撰写报告初稿。
  * **内容改写与扩充:** 将简单的要点扩写成段落，或将现有文本改写成不同风格。
  * **个性化内容生成:** 为不同用户生成定制化的新闻摘要、产品推荐语等。

**案例：使用 Hugging Face `pipeline` 生成故事开头**

Hugging Face 的 `pipeline` 提供了一个非常方便的高级接口，可以快速体验各种预训练模型的功能，包括文本生成。

```python
from transformers import pipeline
import torch

# 检查是否有 GPU 可用
device_id = 0 if torch.cuda.is_available() else -1

# 加载文本生成 pipeline，使用一个中等大小的模型如 gpt2-medium
# 对于更强的能力，可以尝试更大的模型 (如果资源允许)
# 例如 'gpt2-large', 'gpt2-xl', 或其他开源模型如 'EleutherAI/gpt-neo-2.7B'
# 注意：大模型需要更多显存和计算资源
try:
    # 尝试加载一个中等模型
    generator = pipeline('text-generation', model='gpt2-medium', device=device_id)
    print("Using gpt2-medium")
except Exception as e:
    print(f"Failed to load gpt2-medium ({e}), falling back to gpt2...")
    generator = pipeline('text-generation', model='gpt2', device=device_id) # 如果中等模型失败，回退到 gpt2
    print("Using gpt2")


# 定义故事开头
prompt = "在一个被遗忘的古老森林深处，住着一位神秘的守林人。据说，他守护着一个不为人知的秘密，那就是..."

# 使用 pipeline 生成文本
# max_length 控制生成文本的总长度 (包括 prompt)
# num_return_sequences 控制生成多少个不同的序列
# do_sample=True 开启采样，使得生成更多样化
# temperature 控制随机性 (越低越保守，越高越随机)
# top_k / top_p 限制采样范围
outputs = generator(
    prompt,
    max_length=150, # 生成的总长度
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    pad_token_id=generator.tokenizer.eos_token_id # 确保 pad token 正确设置
)

print("\nGenerated Story Opening:")
for i, output in enumerate(outputs):
    print(f"--- Option {i+1} ---")
    # output['generated_text'] 包含 prompt + 生成内容
    print(output['generated_text'])
    print("-" * 20)

# 示例输出可能类似 (每次运行会不同):
# Generated Story Opening:
# --- Option 1 ---
# 在一个被遗忘的古老森林深处，住着一位神秘的守林人。据说，他守护着一个不为人知的秘密，那就是...
# 森林的心脏地带隐藏着一个通往另一个维度的入口。守林人并非人类，而是一位古老的元素精灵，
# 他的职责是防止那个维度中的混乱力量侵入我们的世界。他独自生活了数个世纪，与树木低语，
# 与飞鸟交流，他的存在本身就是一个传奇。直到有一天，一个迷路的年轻探险家偶然闯入了他的领地...
# --------------------
```

**挑战:**

* **一致性与逻辑性:** 长文本生成时保持前后一致和逻辑连贯。
* **事实准确性:** 生成的内容可能包含事实错误（幻觉）。
* **重复性:** 有时会陷入重复生成某些短语或句子的循环。
* **控制性:** 精确控制生成内容的具体细节（如情节走向、人物性格）比较困难。

**10.2 智能问答与信息检索**

大模型可以利用其在预训练阶段吸收的庞大知识库来回答用户提出的问题。

* **应用方式:**

  * **直接问答 (Open-Domain QA):** 对于常识性、事实性的问题，可以直接向模型提问（零样本 Prompt）。模型会尝试从其内部知识库中检索或生成答案。
  * **基于上下文的问答 (Contextual QA / Reading Comprehension):** 给定一段上下文文本和一个关于该文本的问题，模型需要从上下文中找到或推断出答案。BERT 类模型在抽取式问答（答案是原文片段）上表现出色。生成式模型也可以通过 Prompt 实现。
  * **检索增强生成 (Retrieval-Augmented Generation, RAG):** 这是处理**领域特定知识**或需要**最新信息**的问答任务的关键技术（将在第11章实战）。其流程是：
    1. **检索 (Retrieve):** 将用户问题通过一个**嵌入模型 (Embedding Model)** 转换为向量，然后在包含大量文档（切块后也已转换为向量）的**向量数据库 (Vector Database)** 中进行相似度搜索，找到与问题最相关的几个文档片段。
    2. **增强 (Augment):** 将检索到的相关文档片段与原始问题一起，构建一个新的、更丰富的 Prompt。
    3. **生成 (Generate):** 将增强后的 Prompt 输入给大语言模型（生成器），让模型基于提供的上下文信息生成最终答案。
* **典型场景:**

  * **通用知识问答:** 回答类似“珠穆朗玛峰有多高？”或“解释一下什么是量子纠缠？”的问题。
  * **客服机器人:** 回答用户关于产品、服务或政策的常见问题。
  * **企业知识库问答:** 让员工能够通过自然语言查询内部文档、手册、规章制度。
  * **医疗/法律等专业领域问答:** （需结合 RAG 和专业数据）提供专业信息查询。

**案例：构建一个基于文档的问答系统（RAG 概念）**

假设我们有一个包含公司产品手册的文档库。

1. **文档处理与索引:**
   * 将手册分割成小的、有意义的文本块 (Chunks)。
   * 使用嵌入模型（如 `sentence-transformers/all-MiniLM-L6-v2`）计算每个文本块的向量表示。
   * 将文本块及其向量存入向量数据库（如 FAISS, ChromaDB）。
2. **用户提问:** 用户问：“我的 X 型号打印机卡纸了怎么办？”
3. **检索:**
   * 计算问题的向量表示。
   * 在向量数据库中搜索与问题向量最相似的文本块，可能找到关于 X 型打印机卡纸处理步骤的段落。
4. **构建 Prompt:**
   ```text
   请根据以下手册内容，回答用户的问题。

   手册内容：
   "X 型打印机卡纸处理：首先，请关闭打印机电源。然后，小心地打开前盖板，找到卡住的纸张。沿送纸方向缓慢拉出纸张，避免撕裂。如果纸张撕裂，请确保取出所有碎片。最后，关闭盖板，重新开启电源。"
   ... (其他相关片段) ...

   用户问题：
   我的 X 型号打印机卡纸了怎么办？

   答案：
   ```
5. **生成答案:** LLM 基于提供的上下文生成答案，例如：“根据手册，您应该先关闭打印机电源，然后打开前盖板，沿送纸方向缓慢拉出卡住的纸张。请确保取出所有纸张碎片，然后关闭盖板并重启打印机。”

**挑战:**

* **幻觉:** 即使有 RAG，模型有时仍可能忽略提供的上下文而产生幻觉。
* **检索质量:** RAG 的效果高度依赖于检索阶段能否找到真正相关的上下文。检索不准会导致答案错误或不相关。
* **知识更新:** 对于直接问答，模型知识是静态的；对于 RAG，需要定期更新文档库和索引。
* **复杂问题推理:** 模型可能难以整合来自多个检索片段的信息或进行复杂推理。

**10.3 机器翻译 (Machine Translation)**

机器翻译是 NLP 的经典任务，也是 Transformer 最初取得突破的应用。

* **应用方式:**
  * **专用 NMT 模型:** 许多基于 Transformer 的 Encoder-Decoder 模型（如 Google Translate 底层模型, MarianMT, NLLB）是专门为翻译任务设计和训练的。它们通常在大量的平行语料库（源语言和目标语言句子对）上进行训练。
  * **通用大模型 (通过 Prompting):** 大型 Decoder-Only 模型（如 GPT-4, Llama）也能通过零样本或少样本提示完成高质量的翻译，尤其是在常见语言对之间。
* **典型场景:**
  * 网页/文档翻译。
  * 实时语音翻译（结合 ASR 和 TTS）。
  * 跨语言信息检索。
  * 辅助人工翻译。

**案例：对比通用大模型与专用翻译模型的效果（概念性）**

* **使用专用模型 (Hugging Face `pipeline`):**
  ```python
  from transformers import pipeline

  # 加载一个英译法翻译 pipeline (使用 Helsinki-NLP 的 MarianMT 模型)
  translator_en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=device_id)
  text_en = "Large language models are changing the world."
  translation_fr = translator_en_fr(text_en)
  print(f"English: {text_en}")
  print(f"French (Dedicated Model): {translation_fr[0]['translation_text']}")
  # 输出可能: Les grands modèles linguistiques changent le monde.
  ```
* **使用通用大模型 (通过 Prompting):**
  ```python
  # (假设 generator 是一个加载了 GPT 或类似模型的 pipeline)
  prompt_translate = f"Translate the following English text to French:\n\nEnglish: Large language models are changing the world.\nFrench:"
  outputs = generator(prompt_translate, max_length=50, num_return_sequences=1, do_sample=False) # 使用确定性解码
  print(f"French (General LLM via Prompt): {outputs[0]['generated_text'].split('French:')[-1].strip()}")
  # 输出可能与专用模型类似，但质量取决于 LLM 本身的能力
  ```

**挑战:**

* **低资源语言:** 对于训练数据稀疏的语言对，翻译质量可能不高。
* **领域适应:** 通用翻译模型在特定专业领域（如医学、法律）的术语翻译可能不准确。需要领域微调。
* **文化和语境:** 准确传达原文的文化内涵和细微语境仍然困难。
* **长距离依赖:** 保持长篇文档翻译的一致性。

**10.4 文本摘要 (Text Summarization)**

自动生成长文本（如新闻文章、报告、会议记录）的简洁摘要。

* **应用方式:**
  * **抽取式摘要 (Extractive):** 从原文中选择最重要的句子或短语组成摘要。传统方法常用，但 LLM 不太常用此方式。
  * **生成式摘要 (Abstractive):** 理解原文内容后，用模型自己的话生成全新的、更简洁的摘要。这是大模型（特别是 Encoder-Decoder 如 BART, Pegasus 或 Decoder-Only 通过 Prompting）擅长的方式。
* **典型场景:**
  * 新闻摘要生成。
  * 报告/论文摘要。
  * 会议纪要/邮件线索摘要。
  * 产品评论摘要。

**案例：生成式摘要 vs. 抽取式摘要**

* **原文:** "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需进行显式编程。常见的技术包括监督学习、无监督学习和强化学习。近年来，深度学习作为机器学习的一个子领域取得了巨大进展，尤其是在图像识别和自然语言处理方面。"
* **生成式摘要 (LLM 可能生成):** "机器学习是 AI 的一个领域，让计算机通过数据学习，主要技术有监督、无监督和强化学习。深度学习是其热门子领域，在图像和 NLP 方面成果显著。" (更简洁，重新组织语言)
* **抽取式摘要 (传统方法可能选择):** "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需进行显式编程。近年来，深度学习作为机器学习的一个子领域取得了巨大进展，尤其是在图像识别和自然语言处理方面。" (直接抽取句子)

**挑战:**

* **事实一致性:** 生成式摘要必须忠于原文的关键信息，不能捏造或歪曲。
* **重要性判断:** 模型需要准确判断原文中哪些信息是核心、哪些是次要的。
* **摘要长度与信息量的权衡:** 如何在保持简洁的同时保留足够的信息。
* **领域适应:** 通用摘要模型可能无法很好地处理高度专业化的文本。

**10.5 代码生成与辅助编程**

这是大模型最令人兴奋和影响深远的应用之一，极大地改变了软件开发流程。

* **应用方式:**
  * **代码补全 (Code Completion):** 根据当前代码上下文，预测并补全下一行或下一段代码。
  * **代码生成 (Code Generation):** 根据自然语言描述（注释、需求文档）生成完整的函数、类或代码片段。
  * **代码解释 (Code Explanation):** 用自然语言解释一段代码的功能和逻辑。
  * **代码调试与 Bug 查找 (Debugging):** 分析代码，指出潜在错误或提出修复建议。
  * **代码翻译 (Code Translation):** 将代码从一种编程语言转换为另一种。
  * **单元测试生成 (Unit Test Generation):** 为给定的函数或类生成测试用例。
* **模型:** 通常使用在大量代码数据（如 GitHub）上预训练或微调的模型（如 OpenAI Codex, Google AlphaCode, Meta Code Llama）或具备强大代码能力的通用 LLM (如 GPT-4, Claude)。
* **工具:** 这些能力被集成到各种编程辅助工具中，如 GitHub Copilot, Amazon CodeWhisperer, Replit Ghostwriter 等。

**案例：使用 Copilot 类工具或直接让大模型生成代码片段**

* **场景 (IDE 中使用 Copilot):** 开发者写下一行注释 `// function to read a csv file into a pandas dataframe`，Copilot 可能会自动建议出完整的 Python 函数代码：
  ```python
  import pandas as pd

  def read_csv_to_dataframe(file_path):
      """Reads a CSV file into a pandas DataFrame."""
      try:
          df = pd.read_csv(file_path)
          return df
      except FileNotFoundError:
          print(f"Error: File not found at {file_path}")
          return None
      except Exception as e:
          print(f"An error occurred: {e}")
          return None
  ```
* **场景 (直接向 LLM 提问):**
  * 提示: `"用 Python 写一个函数，接收一个整数列表，返回其中所有偶数的平方和。"`
  * LLM 可能生成:
    ```python
    def sum_of_squares_of_evens(numbers):
      """Calculates the sum of the squares of all even numbers in a list."""
      total = 0
      for num in numbers:
        if num % 2 == 0:
          total += num * num
      return total
    ```

**挑战:**

* **正确性与 Bug:** 生成的代码可能包含逻辑错误或难以发现的 Bug。
* **安全性:** 生成的代码可能存在安全漏洞（如 SQL 注入、缓冲区溢出）。
* **效率:** 生成的代码可能不是最优的。
* **可维护性:** 生成的代码风格可能不一致或难以理解。
* **版权与许可:** 使用受特定许可证保护的代码进行训练可能引发法律问题。
* **过度依赖:** 开发者可能过度依赖 AI 而忽视了自身能力的提升和对代码的深入理解。

**10.6 对话系统与聊天机器人**

大模型使得构建更自然、更流畅、更有知识、更能理解上下文的对话系统成为可能。

* **应用方式:**
  * **基础模型:** 使用经过预训练和指令微调的 Decoder-Only 模型作为对话引擎。
  * **上下文管理:** 需要有效的机制来管理多轮对话的上下文历史（通常通过将历史对话包含在 Prompt 中实现，但受限于上下文长度）。
  * **安全与对齐:** 对话模型尤其需要进行严格的对齐训练（如 RLHF, CAI），以确保其回答安全、负责、不产生有害内容。
  * **(可选) 工具使用/Function Calling:** 让模型能够调用外部 API 或工具（如搜索引擎、计算器、日历 API）来获取实时信息或执行操作（见第12章 Agent 部分）。
* **典型场景:**
  * **通用聊天机器人:** 如 ChatGPT, Claude, Google Bard/Gemini。
  * **客服机器人:** 提供更自然的交互体验。
  * **虚拟助手:** 如智能音箱的对话能力。
  * **教育辅导机器人:** 提供个性化的学习对话。
  * **心理陪伴机器人:** 提供情感支持和对话。

**案例：讨论构建流畅、有记忆、安全的聊天机器人的关键技术点**

* **流畅性:** 依赖于基础模型的强大生成能力和在对话数据上的微调。
* **记忆 (上下文理解):**
  * **滑动窗口:** 在 Prompt 中包含最近的几轮对话历史。
  * **摘要机制:** 对较早的对话历史进行摘要，将摘要包含在 Prompt 中。
  * **向量数据库检索:** 将历史对话存储在向量数据库中，根据当前对话内容检索相关历史片段加入 Prompt (类似 RAG)。
* **安全性:**
  * **严格的数据过滤:** 在预训练和微调阶段过滤有害数据。
  * **指令微调:** 使用包含安全指令的数据进行微调（如“请不要生成不安全内容”）。
  * **RLHF/CAI:** 通过人类或 AI 反馈，强化模型的安全行为。
  * **内容过滤器:** 在模型输出前或输出后使用额外的分类器或规则来检测和过滤不当内容。
  * **红队测试:** 主动发现安全漏洞。

**挑战:**

* **长期记忆与上下文丢失:** 如何在极长对话中保持关键信息。
* **一致性:** 保持角色、观点和事实的长期一致性。
* **幻觉:** 在对话中仍可能出现事实错误。
* **过度安全与回避:** 有时为了安全，模型可能过于保守，拒绝回答一些正常的问题。
* **个性化与角色扮演的平衡:** 如何在保持一致角色的同时适应不同用户的需求。

**10.7 情感分析与文本分类**

这是 NLP 的基础任务，用于判断文本的情感倾向（正面、负面、中性）或将其归入预定义的类别（如新闻分类、主题分类）。

* **应用方式:**
  * **微调 (Fine-tuning):** 使用带有标签的数据对 Encoder-Only 模型（如 BERT, RoBERTa）或 Encoder-Decoder/Decoder-Only 模型进行微调（见 6.2.3 案例）。这是传统上获得最佳性能的方式。
  * **零样本/少样本提示 (Zero/Few-shot Prompting):** 对于类别定义清晰的任务，可以直接通过提示让大型 Decoder-Only 模型进行分类，无需微调。例如，可以设计提示：“以下评论的情感是积极还是消极？评论：[评论内容] 情感：”。
* **典型场景:**
  * **产品评论分析:** 了解用户对产品的看法。
  * **社交媒体监控:** 跟踪品牌声誉或公众情绪。
  * **客户反馈分类:** 自动将用户反馈归类（如 Bug 报告、功能请求、投诉）。
  * **新闻主题分类:** 自动将新闻划分到不同版块。

**案例：回顾第6章微调案例，讨论零样本分类能力**

* 我们在 6.2.3 节看到了如何微调 BERT 进行 IMDB 情感分类，通常能达到很高的准确率。
* 对于 GPT-3.5 或 GPT-4 这样的大模型，我们可以尝试零样本分类：
  ```python
  # (假设 generator 是 GPT-3.5/4 的接口或 pipeline)
  prompt_classify = """
  Is the sentiment of the following movie review positive or negative?

  Review: "This movie was a complete waste of time. The acting was terrible and the plot made no sense."

  Sentiment:""" # 注意结尾的 'Sentiment:' 引导模型输出

  outputs = generator(prompt_classify, max_length=len(prompt_classify.split()) + 10, num_return_sequences=1, do_sample=False)
  sentiment = outputs[0]['generated_text'].split('Sentiment:')[-1].strip()
  print(f"Predicted Sentiment (Zero-shot): {sentiment}") # 可能输出 Negative
  ```
* **权衡:** 微调通常更准确、更鲁棒，但需要标注数据和计算资源。零样本提示更快捷、成本低，适用于简单分类或快速原型验证，但性能可能不稳定，且对提示设计敏感。

**10.8 向量嵌入（Embeddings）的应用**

大模型（或专门的嵌入模型）可以将文本（单词、句子、段落、文档）映射到高维向量空间，这些向量（嵌入）捕捉了文本的语义信息。这些嵌入本身就是一种强大的副产品，可以驱动许多下游应用。

* **获取嵌入:**
  * **使用专门的嵌入模型:** 如 Sentence-Transformers 库提供的模型（`all-MiniLM-L6-v2`, `multi-qa-mpnet-base-dot-v1` 等），这些模型专门为生成高质量的句子/段落嵌入而优化。
  * **使用通用大模型:** 取大模型（如 BERT, Llama）某个隐藏层的输出，并通过池化（Pooling，如取 [CLS] Token 的输出，或对所有 Token 输出取平均值）得到整个输入的嵌入。Hugging Face `transformers` 库可以方便地提取这些嵌入。
  * **通过 API 获取:** OpenAI 等服务提供专门的嵌入 API。
* **核心原理:** 语义相似的文本在向量空间中距离更近。距离通常用**余弦相似度 (Cosine Similarity)** 来衡量（计算两个向量夹角的余弦值，范围从 -1 到 1，越接近 1 表示越相似）。

**向量嵌入的典型应用:**

**10.8.1 语义搜索 (Semantic Search)**

* **目标:** 根据查询语句的**语义含义**来搜索相关文档，而不是仅仅基于关键词匹配。
* **流程:**
  1. **离线索引:** 将文档库中的所有文档（或文本块）转换为嵌入向量并存储在向量数据库中（如 FAISS, Milvus, Pinecone, ChromaDB）。
  2. **在线查询:** 将用户的查询语句转换为嵌入向量。
  3. **相似度搜索:** 在向量数据库中搜索与查询向量余弦相似度最高的 K 个文档向量。
  4. **返回结果:** 返回对应的 K 个文档。
* **优势:** 能找到概念相关但用词不同的文档，理解用户的真实意图。是 RAG 的核心组件。

**10.8.2 推荐系统 (Recommendation Systems)**

* **目标:** 向用户推荐他们可能感兴趣的物品（如商品、文章、视频、音乐）。
* **应用嵌入:**
  * **基于内容的推荐:** 计算物品描述的嵌入向量，推荐与用户过去喜欢的物品内容相似的其他物品。
  * **协同过滤的补充:** 结合用户行为（如点击、购买）和物品内容的嵌入信息进行推荐。
  * **用户画像:** 基于用户的历史行为或描述生成用户嵌入，寻找与用户兴趣相似的物品或用户。

**10.8.3 聚类分析 (Clustering)**

* **目标:** 将大量无标签的文本数据自动分组，使得同一组内的文本语义相似，不同组的文本语义不同。
* **流程:**
  1. 计算所有文本的嵌入向量。
  2. 使用聚类算法（如 K-Means, DBSCAN, Hierarchical Clustering）对这些向量进行聚类。
* **应用:** 主题发现、文本分类（无监督）、异常检测。

**代码示例：使用 Sentence Transformers 获取文本嵌入，并进行相似度计算**

```python
from sentence_transformers import SentenceTransformer, util
import torch

# 1. 加载预训练的 Sentence Transformer 模型
#    'all-MiniLM-L6-v2' 是一个速度快且效果不错的常用模型
model_st = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 准备要计算嵌入的句子
sentences = [
    "大语言模型正在改变世界。",
    "人工智能的最新进展令人瞩目。",
    "今天天气真不错，适合散步。",
    "机器学习是实现人工智能的一种方式。",
]

# 3. 计算嵌入向量
print("Calculating embeddings...")
# model.encode() 返回 NumPy 数组或 PyTorch 张量
embeddings = model.encode(sentences, convert_to_tensor=True) # 获取 torch.Tensor

print("Embeddings shape:", embeddings.shape) # (num_sentences, embedding_dim) e.g., (4, 384)

# 4. 计算句子之间的余弦相似度
print("\nCalculating cosine similarity between sentences:")
# 计算第一个句子与其他所有句子的相似度
query_embedding = embeddings[0]
other_embeddings = embeddings[1:]

# 使用 sentence_transformers.util 或 torch.nn.functional 计算
cosine_scores = util.cos_sim(query_embedding, other_embeddings)
# 或者: cosine_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), other_embeddings)

print(f"\nSimilarity between '{sentences[0]}' and:")
for i in range(len(other_embeddings)):
    print(f"- '{sentences[i+1]}': {cosine_scores[0][i]:.4f}")

# 示例输出 (数值会精确):
# Similarity between '大语言模型正在改变世界。' and:
# - '人工智能的最新进展令人瞩目。': 0.6xxx (较高)
# - '今天天气真不错，适合散步。': 0.0xxx (很低)
# - '机器学习是实现人工智能的一种方式。': 0.7xxx (最高)

# 计算所有句子对之间的相似度矩阵
all_pairs_similarity = util.cos_sim(embeddings, embeddings)
print("\nSimilarity matrix between all pairs:")
print(all_pairs_similarity)
```

**挑战:**

* **选择合适的嵌入模型:** 不同模型适用于不同任务（如对称任务 vs. 非对称任务）和语言。
* **嵌入维度与计算/存储成本:** 更高维度的嵌入通常效果更好，但也需要更多存储和计算。
* **领域适应:** 通用嵌入模型在特定专业领域的语义捕捉可能不够精确。
* **向量数据库的选择与优化:** 对于大规模应用，选择和配置高效的向量数据库很重要。

**10.9 本章小结：大模型赋能千行百业**

本章我们巡礼了大语言模型的诸多典型应用场景，展示了它们作为通用人工智能工具的巨大潜力：

* 从**文本生成与创作**，到**智能问答与信息检索**（特别是通过 RAG）；
* 从**机器翻译**和**文本摘要**，到**代码生成与辅助编程**；
* 从构建更智能的**对话系统与聊天机器人**，到执行**情感分析与文本分类**；
* 以及利用核心的**向量嵌入**能力驱动**语义搜索、推荐系统和聚类分析**等下游应用。

这些应用场景远非全部，随着模型能力的不断增强和研究的深入，新的应用正在不断涌现。理解这些典型的应用模式，有助于我们思考如何将大模型技术有效地集成到具体的产品、服务或工作流程中，以解决实际问题，创造新的价值。

在了解了这些广泛的应用之后，下一章 **第11章：实战项目：构建一个领域知识问答机器人**，我们将亲自动手，选择其中一个重要且实用的场景——基于 RAG 的领域知识问答——来构建一个完整的应用，将前面学到的理论和技术知识付诸实践。

---
