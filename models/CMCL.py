import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mamba_ssm import Mamba

class HyperMambaSelector(nn.Module):

    def __init__(
        self,
        dim: int = 4096,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_select: int = 32,      
        hyper_hidden: int = 512,
        dropout: float = 0.0,
        bidirectional: bool = True, 
    ):
        super().__init__()
        self.dim = dim
        self.num_select = num_select
        self.bidirectional = bidirectional

        self.mamba_fwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=0,
        )
        if bidirectional:
            self.mamba_bwd = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=1,
            )


        self.hyper_net = nn.Sequential(
            nn.Linear(dim, hyper_hidden),
            nn.GELU(),
            nn.LayerNorm(hyper_hidden),
            nn.Dropout(dropout),
            nn.Linear(hyper_hidden, dim * 2),  # 生成 dt_bias 和 conv_bias
        )

        self.selector_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
        )

        self.align_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.gate_output = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.gate_class  = nn.Parameter(torch.tensor(0.38, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)

    def _dynamic_mamba(self, x: torch.Tensor, dt_bias: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
      
        out_fwd = self.mamba_fwd(x) + dt_bias + conv_bias

        if self.bidirectional:
            out_bwd = self.mamba_bwd(x.flip(1)).flip(1) + dt_bias + conv_bias
            out = out_fwd + out_bwd
        else:
            out = out_fwd

        return out

    mimic
    def forward(
        self,
        output_tokens: torch.Tensor,      # (B, L_out, D)
        class_feature: torch.Tensor,      # (B, L_cls, D)
        output_mask: Optional[torch.Tensor] = None  # (B, L_out)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_out, D = output_tokens.shape

        class_pooled = class_feature.mean(dim=1)  # (B, D)
        hyper_out = self.hyper_net(class_pooled)  # (B, 2*D)
        dt_bias, conv_bias = hyper_out[:, :D], hyper_out[:, D:]  # (B, D) each

        dt_bias = dt_bias.unsqueeze(1).expand(-1, L_out, -1)
        conv_bias = conv_bias.unsqueeze(1).expand(-1, L_out, -1)

        mamba_out = self._dynamic_mamba(output_tokens, dt_bias, conv_bias)  # (B, L_out, D)

        scores = self.selector_head(mamba_out).squeeze(-1)  # (B, L_out)
        if output_mask is not None:
            scores = scores.masked_fill(output_mask.bool(), float('-inf'))

        k = min(self.num_select, L_out)
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
        topk_tokens = torch.gather(
            mamba_out, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )

        attn_weights = F.softmax(topk_scores / 0.07, dim=1).unsqueeze(-1)  # (B, k, 1)
        selected_repr = (topk_tokens * attn_weights).sum(dim=1, keepdim=True)  # (B, 1, D)
        selected_repr = self.align_proj(selected_repr).expand(-1, L_out, -1)

        aligned_output = output_tokens + self.gate_output * selected_repr
        aligned_output = F.layer_norm(aligned_output, (D,), eps=1e-6)

        output_pooled = aligned_output.mean(dim=1, keepdim=True)
        aligned_class = class_feature + self.gate_class * output_pooled.expand(-1, class_feature.size(1), -1)
        aligned_class = F.layer_norm(aligned_class, (D,), eps=1e-6)

        return aligned_output, aligned_class


# ffair
# class HyperMambaSelector(nn.Module):

#     def __init__(
#         self,
#         dim: int = 512,  
#         d_state: int = 64,
#         d_conv: int = 4,
#         expand: int = 2,
#         num_select: int = 32,        
#         hyper_hidden: int = 512,
#         dropout: float = 0.0,
#         bidirectional: bool = True,  
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_select = num_select
#         self.bidirectional = bidirectional

#         self.mamba_fwd = Mamba(
#             d_model=dim,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             layer_idx=0,
#         )
#         if bidirectional:
#             self.mamba_bwd = Mamba(
#                 d_model=dim,
#                 d_state=d_state,
#                 d_conv=d_conv,
#                 expand=expand,
#                 layer_idx=1,
#             )

#         self.hyper_net = nn.Sequential(
#             nn.Linear(dim, hyper_hidden),
#             nn.GELU(),
#             nn.LayerNorm(hyper_hidden),
#             nn.Dropout(dropout),
#             nn.Linear(hyper_hidden, dim * 2), 
#         )

#         self.selector_head = nn.Sequential(
#             nn.Linear(dim, dim // 4),
#             nn.GELU(),
#             nn.Linear(dim // 4, 1)
#         )

#         self.align_proj = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim),
#             nn.GELU(),
#             nn.Linear(dim, dim)
#         )

#         self.gate_output = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
#         self.gate_class  = nn.Parameter(torch.tensor(0.38, dtype=torch.float32))

#         self.dropout = nn.Dropout(dropout)

#     def _dynamic_mamba(self, x: torch.Tensor, dt_bias: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
#         # x shape: [B, L, D]
#         # biases shape: [B, L, D]
#         out_fwd = self.mamba_fwd(x) + dt_bias + conv_bias

#         if self.bidirectional:
#             out_bwd = self.mamba_bwd(x.flip(1)).flip(1) + dt_bias + conv_bias
#             out = out_fwd + out_bwd
#         else:
#             out = out_fwd

#         return out

#     def forward(
#         self,
#         output_tokens: torch.Tensor,      
#         class_feature: torch.Tensor,      
#         output_mask: Optional[torch.Tensor] = None 
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
    
#         is_seq_first = False
#         if output_tokens.dim() == 3 and class_feature.dim() == 3:
#             if output_tokens.size(1) == class_feature.size(0) and output_tokens.size(0) != class_feature.size(0):
#                 output_tokens = output_tokens.permute(1, 0, 2) # -> [Batch, Seq, Dim]
#                 is_seq_first = True
        
#         B, L_out, D = output_tokens.shape

#         if hasattr(self, 'hyper_net'):
#             class_pooled = class_feature.mean(dim=1) 
#             hyper_out = self.hyper_net(class_pooled)  
#             dt_bias, conv_bias = hyper_out[:, :D], hyper_out[:, D:]
#             dt_bias = dt_bias.unsqueeze(1).expand(-1, L_out, -1)
#             conv_bias = conv_bias.unsqueeze(1).expand(-1, L_out, -1)
#         else:
#             dt_bias = torch.zeros(B, L_out, D, device=output_tokens.device)
#             conv_bias = torch.zeros(B, L_out, D, device=output_tokens.device)

#         mamba_out = self._dynamic_mamba(output_tokens, dt_bias, conv_bias)

#         if hasattr(self, 'selector_head'):
#             scores = self.selector_head(mamba_out).squeeze(-1)
#             if output_mask is not None:
#                 if is_seq_first and output_mask.shape == (L_out, B):
#                      output_mask = output_mask.permute(1, 0)
#                 if output_mask.shape[:2] == (B, L_out):
#                      scores = scores.masked_fill(output_mask.bool(), float('-inf'))
#         else:
#             scores = torch.zeros(B, L_out, device=output_tokens.device)

#         k = min(self.num_select, L_out)
#         topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
#         topk_tokens = torch.gather(mamba_out, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))

#         attn_weights = F.softmax(topk_scores / 0.07, dim=1).unsqueeze(-1)
#         selected_repr = (topk_tokens * attn_weights).sum(dim=1, keepdim=True)
#         selected_repr = self.align_proj(selected_repr)
#         selected_repr_expanded = selected_repr.expand(-1, L_out, -1)

#         aligned_output = output_tokens + self.gate_output * selected_repr_expanded
#         aligned_output = F.layer_norm(aligned_output, (D,), eps=1e-6)

#         output_pooled = aligned_output.mean(dim=1, keepdim=True) 
#         aligned_class = class_feature + self.gate_class * output_pooled
#         aligned_class = F.layer_norm(aligned_class, (D,), eps=1e-6)


#         if is_seq_first:

#             aligned_output = aligned_output.permute(1, 0, 2)
#             aligned_class = aligned_class.permute(1, 0, 2)
        

#         seq_dim = 0 if is_seq_first else 1
#         if aligned_class.size(seq_dim) == 1 and aligned_output.size(seq_dim) != 1:
#             if is_seq_first:
#                 aligned_class = aligned_class.expand(aligned_output.size(0), -1, -1)
#             else:
#                 aligned_class = aligned_class.expand(-1, aligned_output.size(1), -1)
        
#         return aligned_output, aligned_class
