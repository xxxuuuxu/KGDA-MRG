import torch
from torch import nn
import torch.nn.functional as F
from .utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer

# mimic
# class Caption(nn.Module):
#     def __init__(self, backbone, transformer, hidden_dim, vocab_size):
#         super().__init__()
#         self.backbone = backbone
#         self.input_proj = nn.Conv2d(
#             backbone.num_channels, hidden_dim, kernel_size=1)
#         self.transformer = transformer
#         self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

#     def forward(self, samples, target, target_mask, class_feature):
#         if not isinstance(samples, NestedTensor):
#             samples = nested_tensor_from_tensor_list(samples)

#         features, pos = self.backbone(samples)
#         src, mask = features[-1].decompose()

#         assert mask is not None

#         hs, class_feature, aligned_out = self.transformer(self.input_proj(src), mask,
#                               pos[-1], target, target_mask, class_feature)
#         # if isinstance(hs, dict):
#         #     hs = hs["output"]
#         # if isinstance(hs, tuple):
#         #     hs = hs[0]
#         out = self.mlp(hs.permute(1, 0, 2))

#         self.last_class_feature = class_feature
#         self.last_hs = aligned_out
        
#         return out




# ffair

# import torch
# from torch import nn
# import torch.nn.functional as F
# try:
#     from .utils import NestedTensor, nested_tensor_from_tensor_list
# except ImportError:
#     pass
# from .backbone import build_backbone
# from .transformer import build_transformer


# class Caption(nn.Module):
#     def __init__(self, backbone, transformer, hidden_dim, vocab_size):
#         super().__init__()
#         self.backbone = backbone
#         self.input_proj = nn.Conv2d(
#             backbone.num_channels, hidden_dim, kernel_size=1)
#         self.transformer = transformer
#         self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

#     # def forward(self, samples, target, target_mask, class_feature):

#     #     if hasattr(samples, 'tensors'):
#     #         samples = samples.tensors

#     #     features_dict = self.backbone(samples)

#     #     feature_nested = features_dict['0']
#     #     src = feature_nested.tensors
#     #     mask = feature_nested.mask


#     #     if src.dim() == 5:
#     #         bs, seq, c, h, w = src.shape

#     #         src_reshaped = src.view(bs * seq, c, h, w)

#     #         src_proj = self.input_proj(src_reshaped)

#     #         src = src_proj.view(bs, seq, -1, h, w)
#     #     else:
#     #         # 单图情况 (兼容旧逻辑)
#     #         src = self.input_proj(src)

#     #     pos_embed = None 

#     #     hs = self.transformer(
#     #         src=src, 
#     #         mask=mask,
#     #         pos_embed=pos_embed,
#     #         tgt=target, 
#     #         tgt_mask=target_mask, 
#     #         class_feature=class_feature
#     #     )

#     #     if isinstance(hs, tuple):
#     #         hs = hs[0] # 取 hidden state
        
#     #     # hs shape: [Seq_Len_Text, Batch, Hidden_Dim] -> permute -> [Batch, Seq_Len_Text, Hidden_Dim]
#     #     out = self.mlp(hs.permute(1, 0, 2))

#     #     # self.last_class_feature = class_feature
#     #     # self.last_hs = aligned_out 
        
#     #     return out
#     def forward(self, samples, target, target_mask, class_feature):

#         if hasattr(samples, 'tensors'):
#             samples = samples.tensors

#         features, pos = self.backbone(samples)

#         feature_nested = features[-1] 

        
#         src = feature_nested.tensors
#         mask = feature_nested.mask

#         if src.dim() == 5:
#             bs, seq, c, h, w = src.shape
#             src_reshaped = src.view(bs * seq, c, h, w)

#             src_proj = self.input_proj(src_reshaped)
#             src = src_proj.view(bs, seq, -1, h, w)
#         else:
#             src = self.input_proj(src)

#         pos_embed = None 

#         hs = self.transformer(
#             src=src, 
#             mask=mask,
#             pos_embed=pos_embed,
#             tgt=target, 
#             tgt_mask=target_mask, 
#             class_feature=class_feature
#         )

#         if isinstance(hs, tuple):
#             hs = hs[0] 
        
#         # hs shape: [Seq_Len_Text, Batch, Hidden_Dim] -> permute -> [Batch, Seq_Len_Text, Hidden_Dim]
#         out = self.mlp(hs.permute(1, 0, 2))
        
#         return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion