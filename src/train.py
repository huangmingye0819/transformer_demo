# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing
import argparse
from tqdm import tqdm
import os
import random
import numpy as np
import math

# 导入我们的模块
from model import (
    TransformerEncoder, TransformerDecoder, TransformerSeq2Seq
)
from utils import save_model, load_model, plot_curves, count_parameters

def set_seed(seed):
    """设置随机种子以便复现 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_tokenizer(dataset_iterator,pre_tokenizer):
    """为源语言或目标语言训练一个 Tokenizer"""
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizer
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    trainer = WordLevelTrainer(special_tokens=special_tokens)
    
    tokenizer.train_from_iterator(dataset_iterator, trainer=trainer)
    
    # 添加 [SOS] 和 [EOS] 模板
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), 
        pad_token="[PAD]"
    )
    return tokenizer

def process_data(batch_size):
    """
    加载和处理 IWSLT2017 (de-en) 数据集
    """
    print("Loading IWSLT2017 dataset...")
    cache_dir = "./dataset_cache"
    raw_datasets = load_dataset("iwslt2017", "iwslt2017-zh-en", cache_dir=cache_dir,trust_remote_code=True)

    # 1. 训练 Tokenizers
    def get_lang_iterator(dataset, lang):
        for data in dataset:
            yield data['translation'][lang]

    print("Training English (source) tokenizer...")
    src_tokenizer = get_tokenizer(get_lang_iterator(raw_datasets["train"], "en"),pre_tokenizer=Whitespace())
    
    print("Training Chinese (target) tokenizer...")
    tgt_tokenizer = get_tokenizer(get_lang_iterator(raw_datasets["train"], "zh"),pre_tokenizer=BertPreTokenizer())
    # [!! 关键新增代码：保存 Tokenizers !!]
    # 1. 定义保存路径 (我们把它和模型保存在同一个地方)
    tokenizer_save_dir = "checkpoints" 
    
    # 2. 确保 "checkpoints" 目录存在
    os.makedirs(tokenizer_save_dir, exist_ok=True) 

    # 3. 定义两个配置文件的完整路径
    src_tokenizer_path = os.path.join(tokenizer_save_dir, "src_tokenizer.json")
    tgt_tokenizer_path = os.path.join(tokenizer_save_dir, "tgt_tokenizer.json")

    # 4. 保存 Tokenizers
    print(f"Saving source (en) tokenizer to {src_tokenizer_path}...")
    src_tokenizer.save(src_tokenizer_path)
    
    print(f"Saving target (zh) tokenizer to {tgt_tokenizer_path}...")
    tgt_tokenizer.save(tgt_tokenizer_path)
    
    # [!! 新增代码结束 !!]
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")

    # 2. Tokenize 数据集
    def tokenize_function(examples):
        src_batch = [item['en'] for item in examples['translation']]
        tgt_batch = [item['zh'] for item in examples['translation']]
        src_encodings = src_tokenizer.encode_batch(src_batch)
        tgt_encodings = tgt_tokenizer.encode_batch(tgt_batch)
        return {
            "src_ids": [enc.ids for enc in src_encodings],
            "tgt_ids": [enc.ids for enc in tgt_encodings],
        }

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["translation"]
    )
    tokenized_datasets.set_format(type='torch')

    # 3. 创建 DataLoaders (Collate Function)
    def collate_fn(batch):
        src_batch_ids = [item['src_ids'] for item in batch]
        tgt_batch_ids = [item['tgt_ids'] for item in batch]
        
        src_padded = torch.nn.utils.rnn.pad_sequence(src_batch_ids, batch_first=True, padding_value=src_pad_id)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch_ids, batch_first=True, padding_value=tgt_pad_id)
        
        # 关键的 "shifted-right" 操作 (Teacher Forcing)
        # tgt_input: [[SOS, w1, w2, EOS], ...]
        tgt_input = tgt_padded[:, :-1]
        # labels: [[w1, w2, EOS, PAD], ...]
        labels = tgt_padded[:, 1:]
        
        return src_padded, tgt_input, labels

    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, src_vocab_size, tgt_vocab_size, src_pad_id, tgt_pad_id

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src_ids, tgt_input, labels in tqdm(dataloader, desc="Training"):
        src_ids, tgt_input, labels = src_ids.to(device), tgt_input.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(src_ids, tgt_input)
        
        # 计算损失 (忽略 padding)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪 
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src_ids, tgt_input, labels in tqdm(dataloader, desc="Evaluating"):
            src_ids, tgt_input, labels = src_ids.to(device), tgt_input.to(device), labels.to(device)

            outputs = model(src_ids, tgt_input)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # 检查 MPS (Apple Silicon GPU)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    # 否则使用 CPU
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # 1. 加载数据
    data = process_data(args.batch_size)
    train_loader, val_loader, src_vocab_size, tgt_vocab_size, src_pad_id, tgt_pad_id = data
    
    # 2. 初始化模型
    encoder = TransformerEncoder(
        src_vocab_size=src_vocab_size,
        d_model=args.embedding_dim,
        num_heads=args.num_heads,
        d_ff=args.feed_forward_dim,
        num_layers=args.num_layers,
    )
    decoder = TransformerDecoder(
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.embedding_dim,
        num_heads=args.num_heads,
        d_ff=args.feed_forward_dim,
        num_layers=args.num_layers,
    )
    model = TransformerSeq2Seq(
        encoder, decoder, args.embedding_dim, tgt_vocab_size
    ).to(device)
    
    model.set_pad_ids(src_pad_id, tgt_pad_id)
    print(f"Model initialized with {count_parameters(model):,} parameters.")

    # 3. 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # (AdamW 是更好的选择 )
    
    # 关键: 忽略 padding token 的损失
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    def log_lr_change(old_lr):
        """记录学习率变化"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] < old_lr:
                print(f'Learning rate decreased from {old_lr:.6f} to {param_group["lr"]:.6f}')
                return param_group['lr']
        return old_lr

    # 4. 训练循环
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.num_epochs} ---")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        val_perplexity = math.exp(val_loss) # Perplexity (PPL)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val PPL = {val_perplexity:.4f}")
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 检查并记录学习率变化
        current_lr = log_lr_change(current_lr)
        
        # 5. 保存最佳模型 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, args.save_path)

    # 6. 绘制曲线 
    plot_curves(train_losses, val_losses, args.plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Seq2Seq Midterm Assignment")
    
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--feed_forward_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4) # 2层Encoder, 2层Decoder
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=10) # 建议至少10-20
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--save_path", type=str, default="checkpoints/seq2seq_model.pth")
    parser.add_argument("--plot_path", type=str, default="results/seq2seq_loss_curve.png")

    args = parser.parse_args()
    main(args)