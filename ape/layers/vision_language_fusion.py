import torch
import torch.utils.checkpoint as checkpoint

from .fuse_helper import BiAttentionBlock


class VisionLanguageFusion(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        stable_softmax_2d=False,
        clamp_min_for_underflow=True,
        clamp_max_for_overflow=True,
        use_checkpoint=False,
        use_attention_mask_v=False,
    ):
        super(VisionLanguageFusion, self).__init__()
        self.use_checkpoint = use_checkpoint

        # early fusion module
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlock(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            drop_path=drop_path,
            init_values=init_values,
            stable_softmax_2d=stable_softmax_2d,
            clamp_min_for_underflow=clamp_min_for_underflow,
            clamp_max_for_overflow=clamp_max_for_overflow,
            use_attention_mask_v=use_attention_mask_v,
        )

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self.b_attn, v, l, attention_mask_v, attention_mask_l, use_reentrant=False)
        else:
            return self.b_attn(v, l, attention_mask_v, attention_mask_l)

    def extra_repr(self):
        return f"use_checkpoint={self.use_checkpoint}"
