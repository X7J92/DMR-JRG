import torch

def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    logit = logit.log_softmax(dim=-1)


    index = idx.unsqueeze(-1).long().cuda()

    nll_loss = -logit.gather(dim=-1, index=index).squeeze(-1)  # [nb * nw, seq_len]

    smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    epsilon = 1e-9
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / (mask.sum(dim=-1)+epsilon)
    else:
        # [nb * nw, seq_len]
        nll_loss = (nll_loss * weights).sum(dim=-1)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous()


def weakly_supervised_loss_text( words_logit, words_id, words_mask):

    words_logit = words_logit.log_softmax(dim=-1)
    nll_loss = cal_nll_loss(words_logit, words_id, words_mask)
    nll_loss[torch.isnan(nll_loss)] = 0
    nll_loss = nll_loss.mean()
    return nll_loss