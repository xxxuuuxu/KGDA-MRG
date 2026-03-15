import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.utils import get_knowledge
import sys
import os
from models.prompt_enhancer import PromptEnhancer
from models.dynamic_rks_generator import DynamicRKSGenerator
from sklearn.preprocessing import normalize
import numpy as np
import ot
import math
import time
from models.CMCL import HyperMambaSelector

class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, label_list=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, config,
                                          return_intermediate=return_intermediate_dec)

        self.knowledge_prompt = get_knowledge(config.knowledge_prompt_path)

        self.dynamic_rks = DynamicRKSGenerator(rks_dim=config.hidden_dim)
        # self.dynamic_rks = SimpleRKSGenerator(rks_dim=config.hidden_dim, label_list=label_list)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.max_seq_len = 100 
        self.temporal_pos_embed = nn.Embedding(self.max_seq_len, d_model)
        self.src_norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, tgt, tgt_mask, class_feature):
            
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        
        class_feature = class_feature[0]
        batch_key_list = [tuple(item) for item in class_feature]



        class_feature = self.dynamic_rks(batch_key_list, tgt.device)

        
        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device),
                          class_feature=class_feature)
        return hs

        

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, config=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = _get_clones(nn.Linear(config.hidden_dim, config.hidden_dim), config.dec_layers)
        self.fc2 = _get_clones(nn.Linear(config.hidden_dim, config.hidden_dim), config.dec_layers)
        self.fc3 = _get_clones(nn.Linear(config.hidden_dim * 2, config.hidden_dim), config.dec_layers)

        self.fusion_gate = nn.Parameter(torch.zeros(1))


        self.hyper_mamba = HyperMambaSelector(
            dim=config.hidden_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            hyper_hidden=512,
            num_select=32,
            dropout=0.0,
            bidirectional=True
        )


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                class_feature=None):

        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.bool()
        
        if tgt_mask is not None:
            tgt_mask = tgt_mask.bool()

        output = tgt
        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if class_feature is not None:
                # ensure class_feature batch size matches output
                if class_feature.size(0) == 1 and output.size(0) > 1:
                    class_feature = class_feature.expand(output.size(0), -1, -1)
        
                # Optional: you might want to provide masks if available (None is ok)
                token_mask = None
                class_mask = None

                aligned_out, aligned_cls = self.hyper_mamba(output, class_feature)

                aligned_out = output
                aligned_cls = class_feature
                
                output = aligned_out
                class_feature = aligned_cls
                
                output = torch.cat((self.fc1[i](output), self.fc2[i](class_feature)), dim=2)
                output = self.fc3[i](output)


            if self.return_intermediate:
                intermediate.append(self.norm(output))


        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, aligned_cls, aligned_out


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]

        tgt2 = self.self_attn(q, k, value=k, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_transformer(config):
    return Transformer(
        config,
        d_model=config.hidden_dim,
        dropout=config.dropout,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        normalize_before=config.pre_norm,
        return_intermediate_dec=False,
    )







# ---------- start replace block ----------
            # if class_feature is not None:
            #     # ------- 标准化输入格式到 (B, L, D) -------
            #     # 假定 output 的原始 shape 可能是 (S, B, D) 或 (B, S, D)
            #     # 我们把它转换为 batch-first: out_b = (B, L_out, D)
            #     if output.dim() == 3 and output.size(0) != class_feature.size(0) and output.size(1) == class_feature.size(0):
            #         # 可能 output 是 (S, B, D) 而 class_feature 是 (B, Lc, D)
            #         # 这种情况把 output permute -> (B, S, D)
            #         out_b = output.permute(1, 0, 2).contiguous()
            #         permuted_back = True  # 记住后面要 permute 回去
            #     elif output.dim() == 3 and output.size(0) == class_feature.size(0):
            #         # 可能 output 已经是 (B, S, D)
            #         out_b = output.contiguous()
            #         permuted_back = False
            #     elif output.dim() == 2:
            #         # 退化情况 (N, D) 当作 (B=1, L=N, D)
            #         out_b = output.unsqueeze(0)
            #         permuted_back = False
            #     else:
            #         # 兜底处理
            #         out_b = output.contiguous()
            #         permuted_back = False
            
            #     # class_feature 标准化为 (B, L_cls, D)
            #     if class_feature.dim() == 3:
            #         cls_b = class_feature.contiguous()
            #     elif class_feature.dim() == 2:
            #         # (L_cls, D) -> (1, L_cls, D)
            #         cls_b = class_feature.unsqueeze(0).contiguous()
            #     else:
            #         raise RuntimeError(f"Unexpected class_feature dim: {class_feature.dim()}")
            
            #     B_out, L_out, D_out = out_b.size()
            #     B_cls, L_cls, D_cls = cls_b.size()
            
            #     # ------- 确保 dtype/device 一致 -------
            #     if cls_b.device != out_b.device:
            #         cls_b = cls_b.to(out_b.device)
            #     if cls_b.dtype != out_b.dtype:
            #         cls_b = cls_b.to(out_b.dtype)
            
            #     # ------- 如果 class_feature batch 为 1，扩展到与 out_b 的 batch 一致 -------
            #     if B_cls == 1 and B_out > 1:
            #         cls_b = cls_b.expand(B_out, -1, -1)  # (B_out, L_cls, D)
            
            #     # ------- 如果 class_feature 的序列长度不等于 output 的 seq_len，则广播/对齐 -------
            #     # 情形 A: class 特征是 per-sample (L_cls == 1)，我们通常希望把它复制到每个 query 位置
            #     if L_cls == 1 and L_out > 1:
            #         cls_b_expanded = cls_b.expand(-1, L_out, -1)  # (B, L_out, D)
            #     elif L_cls == L_out:
            #         cls_b_expanded = cls_b
            #     else:
            #         # 其他不等长情况：尝试用插值/重复/截断策略 — 这里使用简单的 repeat/trim 策略
            #         if L_cls < L_out:
            #             # 重复 tile 到足够长度再截取
            #             reps = (L_out + L_cls - 1) // L_cls
            #             cls_b_expanded = cls_b.repeat(1, reps, 1)[:, :L_out, :]
            #         else:
            #             # L_cls > L_out：截断前 L_out 个
            #             cls_b_expanded = cls_b[:, :L_out, :]
            
            #     # ------- 调用对齐模块：期望输入 (B, L, D) -------
            #     # cmcr 返回也应该是 (B, L, D) 形式：aligned_out, aligned_cls
            #     aligned_out_b, aligned_cls_b = self.cmcr(out_b, cls_b_expanded)
            
            #     # ------- 将 aligned_* 转回原有 output 的 shape -------
            #     # 如果我们之前 permuted output (S,B,D) -> (B,S,D)，要 permute 回去
            #     if permuted_back:
            #         aligned_out = aligned_out_b.permute(1, 0, 2).contiguous()  # back to (S, B, D)
            #     else:
            #         # 如果原 output 是 (B, S, D) 或其他 batch-first，aligned_out_b 已经符合
            #         aligned_out = aligned_out_b
            
            #     # aligned_cls_b currently shape (B, L_out, D) (因为我们扩展/对齐到 L_out)
            #     # 但原始 class_feature 形状可能不同，需要恢复到原来形态以继续后续使用
            #     # 我们以原始 class_feature 的 batch/seq 为准来还原：
            #     if class_feature.dim() == 3:
            #         # 如果原本是 batch-first 且 batch 与 aligned 匹配，则把 aligned_cls_b 变回 (B_cls, L_cls, D) 形态
            #         if class_feature.size(0) == aligned_cls_b.size(0) and class_feature.size(1) == aligned_cls_b.size(1):
            #             aligned_cls = aligned_cls_b
            #         else:
            #             # 如果原 class_feature batch==1 或 L_cls != L_out，我们把 aligned_cls_b 按需缩回：
            #             if class_feature.size(0) == 1:
            #                 # 原来是 (1, L_cls, D)，取 aligned_cls_b[0, :L_cls, :]
            #                 aligned_cls = aligned_cls_b[0:1, :class_feature.size(1), :].contiguous()
            #             else:
            #                 # 兜底，截取前 L_cls
            #                 aligned_cls = aligned_cls_b[:, :class_feature.size(1), :].contiguous()
            #     else:
            #         # 原来是 (L_cls, D) 形式 -> 返回 (L_cls, D)
            #         aligned_cls = aligned_cls_b[0, :class_feature.size(0), :].contiguous()
            
            #     # ------- 把 output/class_feature 替换成对齐后的版本 -------
            #     output = aligned_out
            #     class_feature = aligned_cls
            
            #     # 如果 class_feature 最终是单 batch (B=1) but downstream expects it expanded, 这里再 expand
            #     if class_feature.dim() == 3 and class_feature.size(0) == 1 and output.size(0) > 1:
            #         class_feature = class_feature.expand(output.size(0), -1, -1)
            
            #     # ------- 接着你的原有拼接操作：注意确保两个张量的 batch 与 seq 长相同 -------
            #     # 如果 output 是 (S, B, D) 且你后续 fc1/2/3 期望 (S,B,D) 结构，请确保你把 fc1/2/3 调用的顺序与 shape 一致
            #     # 假设你的 fc1/2/3 系列期望 batch-first (B, L, D) 或序列优先 (S, B, D)，下面给两种调用示例：
            #     # 这里我们按原代码继续：先调用 fc1[i] 在 output 上，fc2[i] 在 class_feature 上，然后 concat dim=2
            #     # 需要保证这两者在前两维排列一致。
            
            #     # 若 output 原先是 (S,B,D)（即 permuted_back==True），fc 层通常接受 (S,B,D) 或 (B,S,D) 取决于你的实现。
            #     # 我假定 fc1/fc2 能直接接受当前 output 与 class_feature 的形状。
            #     output = torch.cat((self.fc1[i](output), self.fc2[i](class_feature)), dim=2)
            #     output = self.fc3[i](output)
            # ---------- end replace block ----------