# predict.py
# (改编自你提供的脚本，以适配当前项目)

import torch
import torch.nn as nn
from tokenizers import Tokenizer
import argparse
import random
import numpy as np
import os

# 导入我们项目中的模型架构
from src.model import (
    TransformerEncoder, TransformerDecoder, TransformerSeq2Seq
)

def set_seed(seed):
    """设置随机种子以便复现 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _create_pad_mask(seq, pad_idx):
    """
    创建 padding mask
    (seq: [batch_size, seq_len])
    返回: [batch_size, 1, 1, seq_len]
    """
    # (seq == pad_idx) 会得到 [B, L], True 是 pad
    # .unsqueeze(1).unsqueeze(2) 变为 [B, 1, 1, L]
    # 在 Transformer 的 multi-head attention 中,
    # (q_len, k_len) 和 (1, k_len) 可以广播
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def _create_future_mask(seq_len, device):
    """
    创建 look-ahead mask (未来掩码)
    返回: [1, 1, seq_len, seq_len]
    """
    # torch.triu, diagonal=1: 返回上三角矩阵,对角线(含)以下为0
    # [[0, 1, 1],
    #  [0, 0, 1],
    #  [0, 0, 0]]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # .unsqueeze(0).unsqueeze(1) 变为 [1, 1, L, L]
    return mask.unsqueeze(0).unsqueeze(1).to(device)


def translate_sentence(model: TransformerSeq2Seq, 
                         sentence: str, 
                         src_tokenizer: Tokenizer, 
                         tgt_tokenizer: Tokenizer, 
                         device, 
                         max_len: int = 50) -> str:
    """
    执行贪心解码 (Greedy Decode) 来翻译单个句子
    (此函数已适配我们的 Tokenizers 和模型架构)
    """

    model.eval() # 切换到评估模式

    # 1. 分词 (我们的 tokenizer 会自动添加 [SOS] 和 [EOS])
    src_encoding = src_tokenizer.encode(sentence)
    src_indices = src_encoding.ids

    # 2. 转换为 Tensor (batch_size = 1)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    # 3. 创建源掩码 (padding mask)
    # src_mask: [1, 1, 1, src_len]
    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    src_mask = _create_pad_mask(src_tensor, src_pad_id)

    # 4. 计算 Encoder 输出 (只需一次)
    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)

    # 5. 自回归解码 (逐个 token 预测)
    tgt_sos_id = tgt_tokenizer.token_to_id("[SOS]")
    tgt_eos_id = tgt_tokenizer.token_to_id("[EOS]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    
    trg_indices = [tgt_sos_id] # 初始输入是 [SOS]

    for i in range(max_len):
        # trg_tensor: [1, current_len]
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)

        # 6. 创建目标掩码 (padding + future)
        # trg_pad_mask: [1, 1, 1, current_len]
        trg_pad_mask = _create_pad_mask(trg_tensor, tgt_pad_id) 
        
        # trg_future_mask: [1, 1, current_len, current_len]
        trg_future_mask = _create_future_mask(trg_tensor.size(1), device)
        
        # 合并: [1, 1, current_len, current_len]
        # 广播 trg_pad_mask 并与 future_mask 合并
        trg_mask = trg_pad_mask | trg_future_mask

        # 7. 获取 Decoder 输出
        with torch.no_grad():
            decoder_output = model.decoder(trg_tensor, encoder_output, src_mask,trg_mask)
            # decoder_output: [1, current_len, d_model]
            
            # fc_out: [1, current_len, tgt_vocab_size]
            output = model.output_layer(decoder_output)

        # 8. 获取最后一个 token 的预测 (贪心)
        # output.argmax(2) -> [1, current_len]
        # [:, -1] -> [1] (取最后一个词)
        pred_token_idx = output.argmax(2)[:, -1].item()

        trg_indices.append(pred_token_idx)

        # 9. 如果预测到 <eos> 则停止
        if pred_token_idx == tgt_eos_id:
            break
    print(f"DEBUG: Raw output indices: {trg_indices}")
    # 10. 将索引转换回 Token
    # skip_special_tokens=True 会自动移除 [SOS] 和 [EOS]
    trg_translation = tgt_tokenizer.decode(trg_indices, skip_special_tokens=True)

    return trg_translation

def run_translation(args):

    # 自动选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)

    # 1. 加载 Tokenizers
    print("Loading tokenizers...")
    try:
        src_tokenizer = Tokenizer.from_file(args.src_tokenizer_path)
        tgt_tokenizer = Tokenizer.from_file(args.tgt_tokenizer_path)
    except FileNotFoundError:
        print(f"Error: Tokenizer files not found.")
        print(f"Please run train.py first to generate:")
        print(f"  {args.src_tokenizer_path}")
        print(f"  {args.tgt_tokenizer_path}")
        return

    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    
    # 2. 使用加载的配置重新构建模型
    print("Initializing model...")
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
    )
    
    # 这一步很重要，模型内部需要 pad_id 来创建 masks
    # (尽管我们在这个脚本里是外部创建的, 但保留它以防万一)
    model.set_pad_ids(src_pad_id, tgt_pad_id)

    # 3. 加载训练好的模型权重
    print(f"Loading model weights from {args.model_path}")
    try:
        # 我们的 train.py 保存的是一个 dict
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please ensure the model has been trained and saved.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure model parameters in predict.py match train.py.")
        return

    # 4. 执行翻译
    print("模型加载完毕。请输入一个英语句子 (或输入 'q' 退出):")

    while True:
        sentence = input("\n[English]: ")
        if sentence.lower() == 'q':
            break
        
        if not sentence:
            continue

        translation = translate_sentence(
            model, sentence, 
            src_tokenizer, tgt_tokenizer, 
            device, args.max_len
        )
        
        # 修正: [German] -> [Chinese]
        print(f"[Chinese]: {translation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer 翻译脚本 (适配本项目)')
    
    # --- 模型架构参数 (必须与 train.py 一致) ---
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--feed_forward_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    
    # --- 文件路径 ---
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/seq2seq_model.pth',
                        help='训练好的模型权重 (.pth) 文件路径')
    
    parser.add_argument('--src_tokenizer_path', type=str, 
                        default='checkpoints/src_tokenizer.json',
                        help='源语言 (en) tokenizer 配置文件路径')
    
    parser.add_argument('--tgt_tokenizer_path', type=str, 
                        default='checkpoints/tgt_tokenizer.json',
                        help='目标语言 (zh) tokenizer 配置文件路径')

    # --- 推理参数 ---
    parser.add_argument("--max_len", type=int, default=50, 
                        help="Maximum length of the generated translation")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    run_translation(args)