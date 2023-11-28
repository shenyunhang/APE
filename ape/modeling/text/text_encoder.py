import logging
from collections import OrderedDict
from typing import List, Union

import torch
from torch import nn

from .clip_wrapper import build_clip_text_encoder, get_clip_embeddings
from .clip_wrapper_open import build_openclip_text_encoder, get_openclip_embeddings


class TextModel(nn.Module):
    def __init__(
        self,
        model_type,
        model_name,
        model_path,
    ):
        super().__init__()

        self.model_type = model_type
        self.model_name = model_name
        self.model_path = model_path

        if self.model_type == "CLIP":
            self.model = build_clip_text_encoder(model_path, pretrain=True)

        if self.model_type == "OPENCLIP":
            self.model, self.tokenizer = build_openclip_text_encoder(model_name, model_path)

        self.model.eval()

    def forward_text(self, text, prompt="a "):
        if self.model_type == "CLIP":
            return get_clip_embeddings(self.model, text, prompt)

        if self.model_type == "OPENCLIP":
            return get_openclip_embeddings(self.model, self.tokenizer, text, prompt)
