# 🤖 从零开始实现 Transformer

本项目是 "Attention Is All You Need" (Vaswani et al., 2017) 论文中 Transformer 模型的 PyTorch 完整实现。

该实现包含一个完整的 Encoder-Decoder 架构，用于**英中（EN-ZH）机器翻译**任务。项目重点在于深入理解模型的每一个构建模块，并通过一系列对比实验来分析模型行为。

## 目录

- [核心特性](https://www.google.com/search?q=%23-核心特性)
- [项目结构](https://www.google.com/search?q=%23-项目结构)
- [环境配置](https://www.google.com/search?q=%23-环境配置)
- [数据集](https://www.google.com/search?q=%23-数据集)
- [如何运行](https://www.google.com/search?q=%23-如何运行)
  - [训练（复现主要实验）](https://www.google.com/search?q=%231-训练复现主要实验)
  - [评估（测试集）](https://www.google.com/search?q=%232-评估测试集)
- [硬件要求](https://www.google.com/search?q=%23-硬件要求)
- [实验结果与分析](https://www.google.com/search?q=%23-实验结果与分析)
  - [主要结果](https://www.google.com/search?q=%231-主要结果)
  - [对比实验 1](https://www.google.com/search?q=%232-对比实验-1过拟合)
  - [对比实验 2：消融实验（移除位置编码）](https://www.google.com/search?q=%233-对比实验-2消融实验移除位置编码)
- [参考文献](https://www.google.com/search?q=%23-参考文献)

## 🌟 核心特性

- **完整架构**: 实现了完整的 Encoder-Decoder Transformer 架构。
- **核心模块**: 包含以下所有关键组件的从零实现：
  - `ScaledDotProductAttention`
  - `MultiHeadAttention`
  - `PositionwiseFeedForward`
  - `PositionalEncoding` 
  - `EncoderLayer` 和 `DecoderLayer`
  - `Encoder` 和 `Decoder`
  - `TransformerSeq2Seq`
- **掩码机制**:
  - **填充掩码 (Padding Mask)**：在 Encoder 和 Decoder 中忽略 `<pad>` 标记。
  - **前瞻掩码 (Look-ahead Mask)**：用于 Decoder，确保在预测时不会“偷看”未来的词。
- **训练策略**:
  - 实现了标签平滑（Label Smoothing）。
  - 结合 AdamW 优化器和权重衰减（Weight Decay）。
  - 实现了带 Warmup 步数的学习率调度器。
- **可复现性**: 提供了精确的训练命令和随机种子，以保证实验结果的可复现性。

## 📁 项目结构

```
.
├── checkpoints/        # (自动创建) 存放训练好的模型权重 (.pth) 和词表 (.json)
├── data/               # (自动创建) Hugging Face 'datasets' 库的缓存目录
├── results/            # 存放训练曲线图和实验结果
│   ├── successful_loss_curve.png  
│   ├── overfitting_loss.png       
│   └── no_pe_loss.png             
├── scripts/
│   └── run.sh          # 用于启动训练的Shell脚本
├── src/
│   ├── model.py        # Transformer 核心模型架构
│   ├── train.py        # 训练、评估、推理的主脚本
│   └── utils.py        # 绘图、参数统计等辅助函数
├── requirements.txt    # 运行所需的 Python 依赖
└── README.md           # 本文档
```

## ⚙️ 环境配置

1. **克隆本仓库**

   ```
   git clone [https://github.com/huangmingye0819/transformer_demo](https://github.com/huangmingye0819/transformer_demo)
   cd transformer_demo
   ```

2. **创建 Conda 环境 (推荐)**

   ```
   conda create -n transformer python=3.10
   conda activate transformer
   ```

3. 安装依赖

   本项目依赖于 PyTorch 和 Hugging Face 生态系统中的 datasets, tokenizers, 和 torchmetrics。

   ```
   # PyTorch
   # pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   
   # 安装其他依赖
   pip install -r requirements.txt
   ```

   `requirements.txt` 文件内容应包含：

   ```
   torch
   torchmetrics
   datasets
   tokenizers
   matplotlib
   tqdm
   ```

## 📚 数据集

本项目使用 **IWSLT 2017 (en-zh)** 数据集。

你**不需要**手动下载。`src/train.py` 脚本会自动使用 Hugging Face `datasets` 库下载和加载数据集。数据将被缓存到你的 `~/.cache/huggingface/datasets` 目录或项目根目录下的 `data/` 文件夹中。

- **训练集**: 约 230k 句对
- **验证集**: 879 句对

`src/data_loader.py` 中的脚本会自动处理以下任务：

- 训练英文和中文的 `WordLevel` BPE 词表。
- 将数据集 token化 并保存到磁盘，以便快速加载。
- 创建用于训练的 `DataLoader`。

## 🚀 如何运行

所有操作均通过 `src/train.py` 脚本执行。

### 1. 训练（复现主要实验）

这是用于复现报告中模型的精确命令。该命令使用了最佳超参数配置，包括所有正则化技巧。

```
python src/train.py \
    --embedding_dim 256 \
    --num_heads 8 \
    --feed_forward_dim 512 \
    --num_layers 4 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 30 \
    --dropout 0.3 \
    --label_smoothing 0.1 \
    --weight_decay 0.01 \
    --warmup_steps 4000 \
    --seed 42 \
    --save_path "checkpoints/iwslt_en_zh_12M.pth" \
    --plot_path "results/successful_loss_curve.png"
```

**参数说明:**

- `--embedding_dim`: 词维度
- `--num_heads`: 注意力头数
- `--feed_forward_dim`: FFN 内部隐藏层维度
- `--num_layers`: Encoder 和 Decoder 各自的层数
- `--batch_size`: 批量大小
- `--learning_rate`: 最大学习率
- `--num_epochs`: 训练轮数
- `--dropout`: Dropout 比例
- `--label_smoothing`: 标签平滑系数
- `--weight_decay`: AdamW 优化器的权重衰减系数
- `--warmup_steps`: 学习率预热的步数
- `--seed`: 固定随机种子，用于复现
- `--save_path`: 最佳模型的保存路径
- `--plot_path`: 训练/验证损失曲线图的保存路径


### 1. 推理（翻译新句子）

加载模型并翻译你输入的句子：

```
python translate.py 
```

脚本将加载模型，处理输入，并打印出中文翻译结果。但是要注意参数要和train.py保持一致

## 💻 硬件要求

- **GPU**: 强烈推荐使用 NVIDIA GPU。
- **显存 (VRAM)**:
  - 对于上述的12M参数配置（`d_model=128`, `batch_size=32`），训练需要约 **6GB - 8GB** 显存。
  - 对于报告中提到的28M参数配置（`d_model=256`），显存需求会显著增加，建议 **16GB** 以上。
- **CPU/RAM**: 数据预处理在 CPU 上完成，建议至少 16GB 内存以获得流畅体验。

## 📊 实验结果与分析

### 1. 主要结果：

我们最终采用了 28M 参数的轻量级模型，并配合了强有力的正则化组合（Dropout=0.3, 权重衰减=0.01, 标签平滑=0.1）。

从下方的损失曲线可以看出，训练损失（Training Loss）和验证损失（Validation Loss）都在健康下降，且二者之间保持着较小的差距。这表明模型有效学习了翻译任务，同时成功地抑制了过拟合。

*(请确保将你的**成功**训练曲线图命名为 `successful_loss_curve.png` 并放置在 `results/` 目录下)*

### 2. 对S比实验 1：过拟合

为了对比，我们尝试了其他参数的模型。

如下图（`results/overfitting_loss.png`）所示，该模型表现出**严重的过拟合**。训练损失（蓝线）迅速下降，但验证损失（橙线）在第 5 个 Epoch 后便停止下降并开始反弹。这证明了对于 IWSLT 这种规模的数据集，模型容量过大且缺乏足够正则化，会导致模型“背诵”训练数据，而丧失泛化能力。

*(请将你对应的过拟合图命名为 `overfitting_loss.png` 放入 `results/` 文件夹)*

### 3. 对比实验 2：消融实验（移除位置编码）

为了验证位置编码的必要性，我们进行了消融实验：**完全移除了位置编码模块**。

结果如下图（`results/no_pe_loss.png`）所示，模型**完全无法学习**。训练损失和验证损失均在高位（约 2.8-3.4）随机波动，没有任何下降趋势。

这有力地证明了 Transformer 架构本身不具备捕捉序列顺序的能力，它必须依赖外部注入的位置信息（即位置编码）才能理解句子的结构和语法。

*(请将你对应的“无法学习”图命名为 `no_pe_loss.png` 放入 `results/` 文件夹)*

## 📄 参考文献

[1] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares,
Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder
for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.
[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
[3] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,
9(8):1735–1780, 1997.
[4] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.
[5] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified
text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.
[6] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning internal representations
by error propagation. Nature, 323(6088):533–536, 1986.
[7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information
processing systems, 30, 2017.
