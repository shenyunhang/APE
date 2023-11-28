import logging
from collections import OrderedDict
from typing import List, Union

import torch
from torch import nn

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


def build_openclip_text_encoder(open_clip_name, open_clip_model):
    import open_clip

    logger = logging.getLogger(__name__)

    print(open_clip.list_pretrained())
    logger.info("Loading pretrained CLIP " + open_clip_name + " " + open_clip_model)

    model, _, preprocess = open_clip.create_model_and_transforms(
        open_clip_name, pretrained=open_clip_model
    )
    tokenizer = open_clip.get_tokenizer(open_clip_name)

    del model.visual

    model.eval()

    return model, tokenizer


def get_openclip_embeddings(model, tokenizer, vocabulary, prompt="a "):
    model.eval()

    sentences = [prompt + x for x in vocabulary]
    text = tokenizer(sentences).to(model.token_embedding.weight.device)

    with torch.no_grad():
        if len(text) > 10000:
            text_features = torch.cat(
                [
                    model.encode_text(text[: len(text) // 2]),
                    model.encode_text(text[len(text) // 2 :]),
                ],
                dim=0,
            )
        else:
            text_features = model.encode_text(text)

    text_features = text_features.detach().contiguous()

    return text_features
