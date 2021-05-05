from abc import ABC
import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SynFueLoss(Loss):
    def __init__(self, rel_criterion, term_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._term_criterion = term_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, term_logits, rel_logits, term_types, rel_types, term_sample_masks, rel_sample_masks):
        # term loss
        term_logits = term_logits.view(-1, term_logits.shape[-1])
        term_types = term_types.view(-1)
        term_sample_masks = term_sample_masks.view(-1).float()

        term_loss = self._term_criterion(term_logits, term_types)
        term_loss = (term_loss * term_sample_masks).sum() / term_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = term_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = term_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
