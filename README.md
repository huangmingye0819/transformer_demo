# transformer_demo
🤖 从零开始实现 Transformer本项目是 "Attention Is All You Need" (Vaswani et al., 2017) 
论文中 Transformer 模型的 PyTorch 完整实现。该实现包含一个完整的 Encoder-Decoder 架构，用于英中（EN-ZH）机器翻译任务。项目重点在于深入理解模型的每一个构建模块，并通过一系列对比实验来分析模型行为。硬件要求实验结果与分析主要结果：“小而精”模型（12M参数）对比实验 1：过拟合（30M参数 + 弱正则化）对比实验 2：消融实验（移除位置编码）参考文献🌟 核心特性完整架构: 实现了完整的 Encoder-Decoder Transformer 架构。核心模块: 包含以下所有关键组件的从零实现：ScaledDotProductAttentionMultiHeadAttentionPositionwiseFeedForwardPositionalEncoding (使用 sin/cos 函数)EncoderLayer 和 DecoderLayerEncoder 和 Decoder 堆栈TransformerSeq2Seq (最终的封装模型)掩码机制:填充掩码 (Padding Mask)：在 Encoder 和 Decoder 中忽略 <pad> 标记。前瞻掩码 (Look-ahead Mask)：用于 Decoder，确保在预测时不会“偷看”未来的词。训练策略:实现了标签平滑（Label Smoothing）。结合 AdamW 优化器和权重衰减（Weight Decay）。实现了带 Warmup 步数的学习率调度器。可复现性: 提供了精确的训练命令和随机种子，以保证实验结果的可复现性。
📁 项目结构.
├── checkpoints/        # (自动创建) 存放训练好的模型权重 (.pth) 和词表 (.json)
├── data/               # (自动创建) Hugging Face 'datasets' 库的缓存目录
├── results/            # 存放训练曲线图和实验结果
│   ├── successful_loss_curve.png  <- 你的成功实验损失图
│   ├── overfitting_loss.png       <- 你的过拟合对比图
│   └── no_pe_loss.png             <- 你的无位置编码消融实验图
├── scripts/
│   └── run.sh          # (可选) 用于启动训练的Shell脚本
├── src/
│   ├── model.py        # Transformer 核心模型架构 (所有模块)
│   ├── train.py        # 训练、评估、推理的主脚本
│   ├── data_loader.py  # (假设) 数据集处理和 Tokenizer
│   └── utils.py        # (假设) 绘图、参数统计等辅助函数
├── requirements.txt    # 运行所需的 Python 依赖
└── README.md           # 本文档
⚙️ 环境配置克隆本仓库git clone [https://github.com/huangmingye0819/transformer_demo](https://github.com/huangmingye0819/transformer_demo)
cd transformer_demo
创建 Conda 环境 (推荐)conda create -n transformer python=3.10
conda activate transformer
安装依赖本项目依赖于 PyTorch 和 Hugging Face 生态系统中的 datasets, tokenizers, 和 torchmetrics。# PyTorch (请根据你的 CUDA 版本从官网 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 获取命令)
# 例如 (CUDA 11.8):
# pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 安装其他依赖
pip install -r requirements.txt
requirements.txt 文件内容应包含：torch
torchmetrics
datasets
tokenizers
matplotlib
tqdm
📚 数据集本项目使用 IWSLT 2017 (en-zh) 数据集。你不需要手动下载。src/train.py 脚本会自动使用 Hugging Face datasets 库下载和加载数据集。数据将被缓存到你的 ~/.cache/huggingface/datasets 目录或项目根目录下的 data/ 文件夹中。训练集: 约 230k 句对验证集: 879 句对src/data_loader.py 中的脚本会自动处理以下任务：训练英文和中文的 WordLevel BPE 词表。将数据集 token化 并保存到磁盘，以便快速加载。创建用于训练的 DataLoader。🚀 如何运行所有操作均通过 src/train.py 脚本执行。1. 训练（复现主要实验）这是用于复现报告中**“小而精”模型（约1200万参数）**的精确命令。该命令使用了最佳超参数配置，包括所有正则化技巧。python src/train.py \
    --embedding_dim 128 \
    --num_heads 4 \
    --feed_forward_dim 512 \
    --num_layers 3 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --num_epochs 40 \
    --dropout 0.3 \
    --label_smoothing 0.1 \
    --weight_decay 0.01 \
    --warmup_steps 4000 \
    --seed 42 \
    --save_path "checkpoints/iwslt_en_zh_12M.pth" \
    --plot_path "results/successful_loss_curve.png"
参数说明:--embedding_dim: $d_{\text{model}}$--num_heads: 注意力头数--feed_forward_dim: FFN 内部隐藏层维度--num_layers: Encoder 和 Decoder 各自的层数--batch_size: 批量大小--learning_rate: 最大学习率--num_epochs: 训练轮数--dropout: Dropout 比例 (应用于模型各处)--label_smoothing: 标签平滑系数--weight_decay: AdamW 优化器的权重衰减系数--warmup_steps: 学习率预热的步数--seed: 固定随机种子，用于复现--save_path: 最佳模型（基于验证损失）的保存路径--plot_path: 训练/验证损失曲线图的保存路径2. 评估（测试集）如果你想在一个已保存的模型上重新运行评估（计算验证集 BLEU 分数）：python src/train.py \
    --load_model "checkpoints/iwslt_en_zh_12M.pth" \
    --eval_only
3. 推理（翻译新句子）使用 --predict 参数来加载模型并翻译你输入的句子：python src/train.py \
    --load_model "checkpoints/iwslt_en_zh_12M.pth" \
    --predict "Hello, how are you?"
脚本将加载模型，处理输入，并打印出中文翻译结果。💻 硬件要求GPU: 强烈推荐使用 NVIDIA GPU。显存 (VRAM):对于上述的“小而精”（12M参数）配置（d_model=128, batch_size=32），训练需要约 6GB - 8GB 显存。对于报告中提到的“30M参数”模型（d_model=256），显存需求会显著增加，建议 16GB 以上。CPU/RAM: 数据预处理在 CPU 上完成，建议至少 16GB 内存以获得流畅体验。📊 实验结果与分析1. 主要结果：“小而精”模型（12M参数）我们最终采用了 12M 参数的轻量级模型，并配合了强有力的正则化组合（Dropout=0.3, 权重衰减=0.01, 标签平滑=0.1）。从下方的损失曲线可以看出，训练损失（Training Loss）和验证损失（Validation Loss）都在健康下降，且二者之间保持着较小的差距。这表明模型有效学习了翻译任务，同时成功地抑制了过拟合。(请确保将你的成功训练曲线图命名为 successful_loss_curve.png 并放置在 results/ 目录下)2. 对S比实验 1：过拟合（30M参数 + 弱正则化）为了对比，我们尝试了更大的模型（约30M参数，d_model=256）并使用了较弱的正则化（dropout=0.1）。如下图（results/overfitting_loss.png）所示，该模型表现出严重的过拟合。训练损失（蓝线）迅速下降，但验证损失（橙线）在第 5 个 Epoch 后便停止下降并开始反弹。这证明了对于 IWSLT 这种规模的数据集，模型容量过大且缺乏足够正则化，会导致模型“背诵”训练数据，而丧失泛化能力。(请将你对应的过拟合图命名为 overfitting_loss.png 放入 results/ 文件夹)3. 对比实验 2：消融实验（移除位置编码）为了验证位置编码（Positional Encoding）的必要性，我们进行了消融实验：使用 12M 参数的“小而精”模型，但完全移除了位置编码模块。结果如下图（results/no_pe_loss.png）所示，模型完全无法学习。训练损失和验证损失均在高位（约 2.8-3.4）随机波动，没有任何下降趋势。这有力地证明了 Transformer 架构本身不具备捕捉序列顺序的能力，它必须依赖外部注入的位置信息（即位置编码）才能理解句子的结构和语法。(请将你对应的“无法学习”图命名为 no_pe_loss.png 放入 results/ 文件夹)📄 参考文献[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
