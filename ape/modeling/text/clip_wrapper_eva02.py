import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from .eva02_clip import create_model_and_transforms, get_tokenizer


class EVA02CLIP(nn.Module):
    def __init__(
        self,
        clip_model="EVA02-CLIP-B-16",
        cache_dir="EVA02_CLIP_B_psz16_s8B.pt",
        dtype="float32",
        max_batch_size=2560,
        freeze=True,
    ):
        super().__init__()
        self.net, _, _ = create_model_and_transforms(
            clip_model, pretrained=cache_dir, force_custom_clip=True
        )
        self.tokenizer = get_tokenizer(clip_model)

        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        del self.net.visual
        if freeze:
            self.net.eval()
            for name, param in self.net.named_parameters():
                param.requires_grad = False
                param.data = param.data.to(self.dtype)

        self.register_buffer("unused_tensor", torch.zeros(1), False)

        self.text_list_to_feature = {}

        self.max_batch_size = max_batch_size

    @property
    def device(self):
        return self.unused_tensor.device

    def infer_image(self, features):
        x = features["image"][0]
        x = self.net.encode_image(x)
        return x

    @autocast(enabled=False)
    @torch.no_grad()
    def encode_text(self, text_list, cache=False):
        if cache and tuple(text_list) in self.text_list_to_feature:
            return self.text_list_to_feature[tuple(text_list)]

        text_token = self.tokenizer(text_list, context_length=77).to(self.device)

        max_batch_size = self.max_batch_size
        if self.device.type == "cpu" or torch.cuda.mem_get_info(self.device)[0] / 1024**3 < 5:
            max_batch_size = min(256, max_batch_size)
        if len(text_token) > max_batch_size:
            chunck_num = len(text_token) // max_batch_size + 1
            encoder_outputs = torch.cat(
                [
                    self.net.encode_text(
                        text_token[chunck_id * max_batch_size : (chunck_id + 1) * max_batch_size]
                    )
                    for chunck_id in range(chunck_num)
                ],
                dim=0,
            )
        else:
            encoder_outputs = self.net.encode_text(text_token)

        ret = {
            "last_hidden_state_eot": encoder_outputs,
        }

        if cache:
            self.text_list_to_feature[tuple(text_list)] = ret

        return ret

    @autocast(enabled=False)
    @torch.no_grad()
    def forward_text(self, text_list, cache=False):
        if cache and tuple(text_list) in self.text_list_to_feature:
            return self.text_list_to_feature[tuple(text_list)]

        text_token = self.tokenizer(text_list, context_length=77).to(self.device)

        max_batch_size = self.max_batch_size
        if self.device.type == "cpu" or torch.cuda.mem_get_info(self.device)[0] / 1024**3 < 5:
            max_batch_size = min(256, max_batch_size)
        if len(text_token) > max_batch_size:
            chunck_num = len(text_token) // max_batch_size + 1
            encoder_outputs = [
                self.custom_encode_text(
                    text_token[chunck_id * max_batch_size : (chunck_id + 1) * max_batch_size],
                    self.net.text,
                )
                for chunck_id in range(chunck_num)
            ]
            encoder_outputs_x = torch.cat([x for (x, _) in encoder_outputs], dim=0)
            encoder_outputs_xx = torch.cat([xx for (_, xx) in encoder_outputs], dim=0)
        else:
            encoder_outputs_x, encoder_outputs_xx = self.custom_encode_text(
                text_token, self.net.text
            )

        end_token_idx = text_token.argmax(dim=-1)
        attention_mask = end_token_idx.new_zeros(encoder_outputs_xx.size()[:2])
        for i in range(attention_mask.size(0)):
            attention_mask[i, : end_token_idx[i] + 1] = 1

        ret = {
            "end_token_idx": end_token_idx,
            "attention_mask": attention_mask,
            "last_hidden_state": encoder_outputs_xx,
            "last_hidden_state_eot": encoder_outputs_x,
        }

        if cache:
            self.text_list_to_feature[tuple(text_list)] = ret

        return ret

    @autocast(enabled=False)
    @torch.no_grad()
    def custom_encode_text(self, text, m, normalize: bool = False):
        cast_dtype = m.transformer.get_cast_dtype()

        x = m.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + m.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = m.transformer(x, attn_mask=m.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = m.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        xx = x @ m.text_projection

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ m.text_projection

        return (
            F.normalize(x, dim=-1) if normalize else x,
            F.normalize(xx, dim=-1) if normalize else xx,
        )
