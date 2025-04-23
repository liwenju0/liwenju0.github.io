---
layout: post
title:  "MHA, GQA, MQA, MLA的代码"
date:   2025-04-01 08:54:08 +0800
category: "Transformer"
published: true
---

本文汇总这几个常见注意力结构的源码，尽可能展示出依次递进的演变过程，以备复习。

<!--more-->

## MHA
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.shape[0]
        
        # 线性变换并分头
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数 (缩放点积注意力)
        # [batch_size, num_heads, query_len, head_dim] x [batch_size, num_heads, head_dim, key_len]
        # -> [batch_size, num_heads, query_len, key_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码（如果提供）
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        if key_padding_mask is not None:
            # 扩展key_padding_mask到合适的维度
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
        
        # softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出 [batch_size, num_heads, query_len, key_len] x [batch_size, num_heads, value_len, head_dim]
        # -> [batch_size, num_heads, query_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置并重新形状化 [batch_size, num_heads, query_len, head_dim] -> [batch_size, query_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 最终的线性变换
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class FeedForward(nn.Module):
    """前馈神经网络，包含两个线性层，中间有激活函数"""
    def __init__(self, embed_dim, ffn_dim, dropout=0.0, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 支持不同的激活函数
        self.activation_name = activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu' or activation == 'swish':
            self.activation = F.silu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层，包含多头注意力和前馈神经网络"""
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 ffn_dim=2048, 
                 dropout=0.1, 
                 activation='relu',
                 norm_first=False):
        super().__init__()
        
        # 多头注意力
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 前馈神经网络
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, activation)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 是否先进行归一化（Pre-LN）还是后归一化（Post-LN）
        self.norm_first = norm_first
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 是否先归一化取决于norm_first参数
        if self.norm_first:
            # Pre-LN
            attn_output = self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + attn_output
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-LN (原始Transformer架构)
            attn_output = self._sa_block(src, src_mask, src_key_padding_mask)
            src = self.norm1(src + attn_output)
            src = self.norm2(src + self._ff_block(src))
        
        return src
    
    # 自注意力块
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, _ = self.self_attn(x, x, x, attn_mask, key_padding_mask)
        return self.dropout1(x)
    
    # 前馈网络块
    def _ff_block(self, x):
        x = self.ffn(x)
        return self.dropout2(x)

```

## GQA
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_kv_heads=None):  # 添加num_kv_heads参数
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # GQA相关修改：添加num_kv_heads参数处理
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert self.num_heads % self.num_kv_heads == 0, "num_heads必须能被num_kv_heads整除"
        # 每个KV头服务的Q头数量
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        # 定义线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # 修改K和V的投影维度为num_kv_heads * head_dim
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 分头，q保持原来的头数，k和v使用较少的头数
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        
        # 转置维度
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        
        # GQA核心修改：扩展k和v以匹配q的头数
        # 每个kv头会被复制num_queries_per_kv次以匹配query头的数量
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # 重复扩展k
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)  # 重复扩展v
        
        # 计算注意力分数 (缩放点积注意力)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码（如果提供）
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
        
        # softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置并重新形状化
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 最终的线性变换
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class FeedForward(nn.Module):
    """前馈神经网络，包含两个线性层，中间有激活函数"""
    def __init__(self, embed_dim, ffn_dim, dropout=0.0, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 支持不同的激活函数
        self.activation_name = activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu' or activation == 'swish':
            self.activation = F.silu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层，包含多头注意力和前馈神经网络"""
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 ffn_dim=2048, 
                 dropout=0.1, 
                 activation='relu',
                 norm_first=False,
                 num_kv_heads=None):  # 添加num_kv_heads参数
        super().__init__()
        
        # 多头注意力，传入num_kv_heads参数
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, num_kv_heads)
        
        # 前馈神经网络
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, activation)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 是否先进行归一化（Pre-LN）还是后归一化（Post-LN）
        self.norm_first = norm_first
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 是否先归一化取决于norm_first参数
        if self.norm_first:
            # Pre-LN
            attn_output = self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + attn_output
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-LN (原始Transformer架构)
            attn_output = self._sa_block(src, src_mask, src_key_padding_mask)
            src = self.norm1(src + attn_output)
            src = self.norm2(src + self._ff_block(src))
        
        return src
    
    # 自注意力块
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, _ = self.self_attn(x, x, x, attn_mask, key_padding_mask)
        return self.dropout1(x)
    
    # 前馈网络块
    def _ff_block(self, x):
        x = self.ffn(x)
        return self.dropout2(x)
```
MQA的实现，与GQA类似，只是将num_kv_heads设置为1。

## MLA
```python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # 计算RMS Norm
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算位置编码的复数表示"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    """应用旋转位置编码"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)

class MLA(nn.Module):
    """多头潜在注意力（Multi-head Latent Attention）"""
    def __init__(self, 
                 dim, 
                 n_heads, 
                 qk_nope_head_dim=128,  # 非位置编码部分的头维度
                 qk_rope_head_dim=64,   # 位置编码部分的头维度
                 v_head_dim=128,        # 值的头维度
                 q_lora_rank=0,         # 查询的低秩投影维度
                 kv_lora_rank=512,      # 键值的低秩投影维度
                 dropout=0.0,
                 attn_impl="naive"):    # 注意力实现方式：naive或absorb
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.attn_impl = attn_impl
        
        # 计算注意力缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # 针对查询的投影层 - 可选使用低秩投影
        if self.q_lora_rank == 0:
            # 直接投影
            self.wq = nn.Linear(dim, n_heads * self.qk_head_dim)
        else:
            # 使用低秩投影 (LoRA)
            self.wq_a = nn.Linear(dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, n_heads * self.qk_head_dim)
        
        # 键值使用低秩投影
        self.wkv_a = nn.Linear(dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # 输出投影
        self.wo = nn.Linear(n_heads * self.v_head_dim, dim)
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, start_pos, freqs_cis, attention_mask=None, max_seq_len=4096):
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        
        # 生成q向量 - 可选使用低秩投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # 将q分为非位置编码部分和位置编码部分
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转位置编码到q_pe
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # 生成kv向量 - 使用低秩投影
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # 使用naive方式实现注意力
        if self.attn_impl == "naive":
            # 完整的query向量
            q = torch.cat([q_nope, q_pe], dim=-1)
            
            # 通过低秩投影生成键值
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            
            # 完整的key向量
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
            
            # 创建KV缓存
            k_cache = torch.zeros(batch_size, max_seq_len, self.n_heads, self.qk_head_dim, device=x.device, dtype=x.dtype)
            v_cache = torch.zeros(batch_size, max_seq_len, self.n_heads, self.v_head_dim, device=x.device, dtype=x.dtype)
            
            # 更新KV缓存
            k_cache[:, start_pos:end_pos] = k
            v_cache[:, start_pos:end_pos] = v
            
            # 计算注意力分数
            scores = torch.einsum("bshd,bthd->bsht", q, k_cache[:, :end_pos]) * self.softmax_scale
            
        else:  # absorb方式实现注意力
            # 获取wkv_b权重
            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
            
            # 计算q_nope与权重的点积
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # 创建KV缓存
            kv_cache = torch.zeros(batch_size, max_seq_len, self.kv_lora_rank, device=x.device, dtype=x.dtype)
            pe_cache = torch.zeros(batch_size, max_seq_len, self.qk_rope_head_dim, device=x.device, dtype=x.dtype)
            
            # 更新KV缓存
            kv_cache[:, start_pos:end_pos] = self.kv_norm(kv)
            pe_cache[:, start_pos:end_pos] = k_pe.squeeze(2)
            
            # 计算注意力分数 - 分别计算非位置编码部分和位置编码部分
            scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_cache[:, :end_pos]) + 
                     torch.einsum("bshr,btr->bsht", q_pe, pe_cache[:, :end_pos])) * self.softmax_scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores += attention_mask.unsqueeze(1)
        
        # 注意力权重计算
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        if self.attn_impl == "naive":
            output = torch.einsum("bsht,bthd->bshd", attn_weights, v_cache[:, :end_pos])
        else:
            # 先与kv_cache相乘
            output = torch.einsum("bsht,btc->bshc", attn_weights, kv_cache[:, :end_pos])
            # 再与权重相乘生成最终输出
            output = torch.einsum("bshc,hdc->bshd", output, wkv_b[:, -self.v_head_dim:])
        
        # 重塑并投影到原始维度
        output = output.reshape(batch_size, seq_len, -1)
        return self.wo(output)

class MLABlock(nn.Module):
    """包含MLA注意力机制的Transformer块"""
    def __init__(self, 
                 dim=768, 
                 n_heads=12, 
                 qk_nope_head_dim=128, 
                 qk_rope_head_dim=64, 
                 v_head_dim=128,
                 q_lora_rank=0, 
                 kv_lora_rank=512, 
                 mlp_ratio=4, 
                 dropout=0.1, 
                 attn_impl="naive"):
        super().__init__()
        
        # 注意力层
        self.attention = MLA(
            dim=dim,
            n_heads=n_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            dropout=dropout,
            attn_impl=attn_impl
        )
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
    def forward(self, x, start_pos, freqs_cis, attention_mask=None):
        # 残差连接 + 注意力层
        x = x + self.attention(self.norm1(x), start_pos, freqs_cis, attention_mask)
        # 残差连接 + 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x

# 简单使用示例
def mla_example():
    # 设置参数
    batch_size = 2
    seq_len = 16
    dim = 512
    n_heads = 8
    max_seq_len = 4096
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 计算位置编码
    freqs_cis = precompute_freqs_cis(64, max_seq_len)
    freqs_cis = freqs_cis[:seq_len]
    
    # 创建注意力掩码 (因果掩码)
    mask = torch.full((seq_len, seq_len), float('-inf')).triu_(1)
    
    # 创建MLA块
    mla_block = MLABlock(
        dim=dim,
        n_heads=n_heads,
        attn_impl="absorb"  # 使用absorb实现
    )
    
    # 前向传播
    output = mla_block(x, start_pos=0, freqs_cis=freqs_cis, attention_mask=mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    return output

```



