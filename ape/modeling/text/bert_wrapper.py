import torch
from torch import nn
from torch.cuda.amp import autocast

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
)


class Bert(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        dtype="float32",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dtype = getattr(torch, dtype)

        self.config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.bert_model = BertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            add_pooling_layer=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        self.bert_model.eval()
        for name, param in self.bert_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.to(self.dtype)

        self.register_buffer("unused_tensor", torch.zeros(1), False)

        self.text_list_to_feature = {}

    @property
    def device(self):
        return self.unused_tensor.device

    @autocast(enabled=False)
    @torch.no_grad()
    def forward_text(self, text_list, cache=False):

        if cache and tuple(text_list) in self.text_list_to_feature:
            return self.text_list_to_feature[tuple(text_list)]

        tokenized = self.tokenizer.batch_encode_plus(
            text_list,
            max_length=256,
            padding="max_length" if True else "longest",
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        input_ids = tokenized.input_ids  # (bs, seq_len)
        attention_mask = tokenized.attention_mask  # (bs, seq_len)

        max_batch_size = 500
        if len(input_ids) > max_batch_size:
            chunck_num = len(input_ids) // max_batch_size + 1
            outputss = [
                self.bert_model(
                    input_ids=input_ids[
                        chunck_id * max_batch_size : (chunck_id + 1) * max_batch_size
                    ],
                    attention_mask=attention_mask[
                        chunck_id * max_batch_size : (chunck_id + 1) * max_batch_size
                    ],
                )
                for chunck_id in range(chunck_num)
            ]

            last_hidden_state = torch.cat(
                [outputs.last_hidden_state for outputs in outputss], dim=0
            )
        else:
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            last_hidden_state = outputs.last_hidden_state

        end_token_idx = input_ids.argmin(dim=-1) - 1

        ret = {
            "end_token_idx": end_token_idx,
            "attention_mask": attention_mask,
            "last_hidden_state": last_hidden_state,
        }

        if cache:
            self.text_list_to_feature[tuple(text_list)] = ret

        return ret
