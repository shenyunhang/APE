import torch


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def reduce_language_feature(features, mask, reduce_type="average"):
    """average pooling of language features""" ""
    if reduce_type == "average":
        embedded = (
            features * mask.unsqueeze(-1).float()
        )  # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float() + 1e-6)
    elif reduce_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0)  # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0)  # (bs, C)
    elif reduce_type == "last":
        out = []
        for i in range(len(features)):
            pool_feat = features[i][torch.argmin(mask[i]) - 1]  # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0)  # (bs, C)
    else:
        raise ValueError("reduce_type should be average or max or last.")
    return aggregate
