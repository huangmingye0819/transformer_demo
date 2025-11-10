# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 模块 1: 位置编码  ---
class PositionalEncoding(nn.Module):
    """
    实现作业中要求的正弦位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return x

# --- 模块 2: 多头注意力 (MHA)  ---
class MultiHeadAttention(nn.Module):
    """
    实现多头自注意力 [cite: 47-49]
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        实现缩放点积注意力 [cite: 44-48]
        """
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        x, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(x)

# --- 模块 3: 前馈网络 (FFN)  ---
class PositionwiseFeedForward(nn.Module):
    """
    实现 Position-wise FFN [cite: 50]
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# --- 模块 4: Encoder 块  ---
class EncoderLayer(nn.Module):
    """
    实现一个 Encoder Block (MHA + FFN + Add & Norm) 
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 残差 + Layer Norm 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output)) # Add & Norm [cite: 52]
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output)) # Add & Norm [cite: 52]
        return x

# --- 模块 5: Transformer Encoder ---
class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1) # [B, L, D] -> [L, B, D] for PositionalEncoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = x.transpose(0, 1) # [L, B, D] -> [B, L, D]
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x # shape: [batch_size, src_seq_len, d_model]

# --- 模块 6: Decoder 块 ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 1. Masked Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 2. Cross-Multi-Head Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # 3. Feed Forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x: [B, tgt_L, D]
        # encoder_output: [B, src_L, D]
        # src_mask: [B, 1, 1, src_L]
        # tgt_mask: [B, 1, tgt_L, tgt_L] (Padding + Future mask)
        
        # 1. Masked Self-Attention (Q,K,V from decoder)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. Cross-Attention (Q from decoder, K,V from encoder)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 3. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x

# --- 模块 7: Transformer Decoder ---
class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1) # [B, L, D] -> [L, B, D]
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = x.transpose(0, 1) # [L, B, D] -> [B, L, D]
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        return x # shape: [batch_size, tgt_seq_len, d_model]

# --- 模块 8: 完整的 Seq2Seq Transformer ---
class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, d_model, tgt_vocab_size):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self.src_pad_id = 0 
        self.tgt_pad_id = 0 
        
    def set_pad_ids(self, src_pad_id, tgt_pad_id):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def create_src_mask(self, src_ids):
        """
        创建源序列的 padding mask [cite: 60]
        """
        # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        mask = (src_ids != self.src_pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    def create_tgt_mask(self, tgt_ids):
        """
        创建目标序列的 padding mask 和 future mask [cite: 60]
        """
        # 1. Padding Mask
        # [batch_size, 1, 1, tgt_seq_len]
        padding_mask = (tgt_ids != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)
        
        # 2. Future Mask (Look-Ahead Mask)
        seq_len = tgt_ids.shape[1]
        device = tgt_ids.device
        # [tgt_seq_len, tgt_seq_len]
        future_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        # [1, 1, tgt_seq_len, tgt_seq_len]
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)
        
        # 3. 组合
        # [batch_size, 1, tgt_seq_len, tgt_seq_len]
        combined_mask = padding_mask & future_mask
        return combined_mask

    def forward(self, src_ids, tgt_ids):
        # src_ids: [B, src_L]
        # tgt_ids: [B, tgt_L] (shifted-right)
        
        # 1. 创建 masks
        src_mask = self.create_src_mask(src_ids) 
        tgt_mask = self.create_tgt_mask(tgt_ids)
        
        # 2. Encoder
        encoder_output = self.encoder(src_ids, src_mask)
        
        # 3. Decoder
        decoder_output = self.decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
        
        # 4. 最终输出
        output = self.output_layer(decoder_output)
        
        return output # [B, tgt_L, tgt_vocab_size]