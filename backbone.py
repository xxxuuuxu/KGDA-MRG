import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from .utils import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(config):
    position_embedding = build_position_encoding(config)
    train_backbone = config.lr_backbone > 0
    return_interm_layers = False
    backbone = Backbone(config.backbone, train_backbone, return_interm_layers, config.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


# ffair
    
# import torch
# import torch.nn.functional as F
# import torchvision
# from torch import nn
# from torchvision.models._utils import IntermediateLayerGetter
# from typing import Dict, List

# try:
#     from .utils import NestedTensor, is_main_process
# except ImportError:
#     from .utils import is_main_process
#     class NestedTensor:
#         def __init__(self, tensors, mask=None):
#             self.tensors = tensors
#             self.mask = mask

# from .position_encoding import build_position_encoding

# class FrozenBatchNorm2d(torch.nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.
#     """
#     def __init__(self, n):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, x):
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = 1e-5
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias


# class BackboneBase(nn.Module):
#     def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
#         super().__init__()

#         for name, parameter in backbone.named_parameters():
#             if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#                 parameter.requires_grad_(False)
        
#         if return_interm_layers:
#             return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#         else:
#             return_layers = {'layer4': "0"}
            
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         self.num_channels = num_channels

#     def forward(self, tensor_input):

#         if isinstance(tensor_input, NestedTensor):
#             x = tensor_input.tensors
#             mask = tensor_input.mask
#         else:
#             x = tensor_input
#             mask = None

#         is_sequence = False
#         if x.dim() == 5:
#             is_sequence = True
#             b, s, c, h, w = x.shape
#
#             x = x.view(b * s, c, h, w)

#         # 进 ResNet
#         xs = self.body(x)
        
#         out: Dict[str, NestedTensor] = {}
#         for name, x_feat in xs.items():
#             # x_feat shape: (B*S, 512, 16, 16)

#             if is_sequence:
#                 _, c_feat, h_feat, w_feat = x_feat.shape
#                 x_feat = x_feat.view(b, s, c_feat, h_feat, w_feat)


#             if mask is not None:

#                  m = F.interpolate(mask[None].float(), size=x_feat.shape[-2:]).to(torch.bool)[0]
#             else:
#                 # 创建全 0 mask (Batch, Seq, H_feat, W_feat) 或 (Batch, H_feat, W_feat)
#                 mask_shape = x_feat.shape[:-3] + x_feat.shape[-2:] # (B, S, H', W')
#                 m = torch.zeros(mask_shape, dtype=torch.bool, device=x_feat.device)

#             out[name] = NestedTensor(x_feat, m)
            
#         return out


# class Backbone(BackboneBase):
#     """ResNet backbone with custom weight loading."""

#     # def __init__(self, name: str,
#     #              train_backbone: bool,
#     #              return_interm_layers: bool,
#     #              dilation: bool):
        
          #     backbone = getattr(torchvision.models, name)(
#     #         replace_stride_with_dilation=[False, False, dilation],
#     #         pretrained=False, norm_layer=FrozenBatchNorm2d)

#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool):

#         if name in ['resnet18', 'resnet34']:
#             replace_stride_with_dilation = [False, False, False]
#         else:
#             replace_stride_with_dilation = [False, False, dilation]

#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=replace_stride_with_dilation,
#             pretrained=False, norm_layer=FrozenBatchNorm2d)
        

        
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

#         weight_path = "/root/autodl-tmp/KGDA-MRG/ADM/weights/FFA_IR_best_weight.pth"
        
#         try:
#             print(f"Visual Extractor: Loading custom weights from {weight_path} ...")
#             state_dict = torch.load(weight_path, map_location="cpu")
            
#             # 清洗权重 Key (移除 fc 层，处理 module. 前缀)
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if "fc" in k: continue # 不需要分类层
#                 k = k.replace("module.", "")
#                 new_state_dict[k] = v
                
#             msg = backbone.load_state_dict(new_state_dict, strict=False)
#             print(f"Weight loading result: {msg}")
            
#         except FileNotFoundError:
#             print(f"Warning: {weight_path} not found! Using random initialization.")
#         except Exception as e:
#             print(f"Error loading weights: {e}")

#         super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding):
#         super().__init__(backbone, position_embedding)

#     def forward(self, tensor_list: NestedTensor):

        
#         xs = self[0](tensor_list) # 调用 BackboneBase.forward
        
#         out: List[NestedTensor] = []
#         pos = []
        
#         for name, x in xs.items():
#             out.append(x)开
#             if x.tensors.dim() == 5:
#                 b, s, c, h, w = x.tensors.shape
#                 x_flat = NestedTensor(x.tensors.view(b*s, c, h, w), x.mask.view(b*s, h, w))
#                 p = self[1](x_flat).to(x.tensors.dtype) # (B*S, C, H, W)
#                 p = p.view(b, s, c, h, w) # 还原
#             else:
#                 p = self[1](x).to(x.tensors.dtype)
                
#             pos.append(p)

#         return out, pos


# def build_backbone(config):
#     position_embedding = build_position_encoding(config)
#     train_backbone = config.lr_backbone > 0
#     return_interm_layers = False
    
#     backbone_name = 'resnet34' 
    
#     backbone = Backbone(backbone_name, train_backbone, return_interm_layers, config.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model