import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import padding, unpadding
from timm.models.helpers import load_custom_pretrained
from timm.models.vision_transformer import default_cfgs

from . import vit
from .decoder import MaskTransformer

class Segmenter(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes,
        pretrained=True
    ):
        super().__init__()
        self.n_cls = num_classes
        self.encoder = getattr(vit, backbone)(n_cls=self.n_cls)
        self.patch_size = self.encoder.patch_size
        self.decoder = MaskTransformer(n_cls=self.n_cls, patch_size=self.patch_size, d_encoder=self.encoder.d_model, n_layers=4, n_heads=self.encoder.d_model//64, d_model=self.encoder.d_model,
                                        d_ff=self.encoder.d_ff, drop_path_rate=0.1, dropout=0.0)
        if pretrained:
            default_cfg = default_cfgs["vit_base_patch16_384"]
            default_cfg["input_size"] = (3, 512, 512)
            load_custom_pretrained(self.encoder, default_cfg)
         
    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def get_param_groups(self):
        param_groups = [[], [], []]
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)
        return param_groups
    
    def get_masks(self):
        return self.masks
    
    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        x = self.encoder(im, return_feature=False)
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, (H, W))
        self.masks = masks
        patches, cls_seg_feat, cls_token = self.decoder.return_feats()
        
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))
        
        return masks, patches, cls_seg_feat, cls_token
