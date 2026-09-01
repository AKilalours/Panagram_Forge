"""Offline smoke harness: exercise the entire training path with no downloads.

Why this exists. The expensive failures in a training pipeline are almost never in the
model. They are in the plumbing: a collator that pads labels wrong, an alignment that
puts a real label on a special token, an evaluation that computes a threshold on the
wrong axis, a checkpoint that does not restore. All of those are findable on a CPU in
under a minute, and every one of them costs a GPU-hour and a confusing result if found
later.

So smoke mode replaces the tokenizer and the backbone with tiny local stand-ins and runs
the real data path, the real loss, the real optimizer step, the real evaluation and the
real checkpoint round-trip. It needs no network and no GPU.

What it does NOT tell you: anything about detection quality. The stand-in encoder is four
attention heads over hashed token ids. A good AUROC here means the plumbing works and the
task is separable, nothing more.
"""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\S+")


class SmokeTokenizer:
    """Hash-based tokenizer with real offset mappings.

    Offsets are what the alignment code consumes, so they have to be genuine character
    spans rather than placeholders, or smoke mode would skip the one thing most likely to
    be wrong.
    """

    pad_token_id = 0
    cls_id = 1
    sep_id = 2

    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size

    def _tid(self, w: str) -> int:
        h = int.from_bytes(hashlib.blake2b(w.lower().encode(), digest_size=4).digest(), "big")
        return 3 + (h % (self.vocab_size - 3))

    def __call__(self, text: str, return_offsets_mapping: bool = True,
                 add_special_tokens: bool = True, truncation: bool = False, **kw) -> dict:
        ids, offs = [], []
        if add_special_tokens:
            ids.append(self.cls_id)
            offs.append((0, 0))          # special tokens carry no text, exactly like a real one
        for m in _WORD.finditer(text):
            ids.append(self._tid(m.group()))
            offs.append((m.start(), m.end()))
        if add_special_tokens:
            ids.append(self.sep_id)
            offs.append((0, 0))
        return {"input_ids": ids, "offset_mapping": offs,
                "attention_mask": [1] * len(ids)}


def build_smoke_model(vocab_size: int = 4096, hidden: int = 64, layers: int = 2):
    """A tiny encoder with the same interface as ForgeBase, built with plain torch."""
    import torch
    from torch import nn

    from forge.modeling.encoder import N_DOC_CLASSES, N_TOKEN_CLASSES

    class SmokeForge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden)
            self.pos = nn.Embedding(1024, hidden)
            enc = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                batch_first=True, dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.doc_head = nn.Linear(hidden, N_DOC_CLASSES)
            self.token_head = nn.Linear(hidden, N_TOKEN_CLASSES)
            self.token_loss_weight = 0.5

        def forward(self, input_ids, attention_mask, doc_labels=None, token_labels=None):
            b, t = input_ids.shape
            pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
            h = self.emb(input_ids) + self.pos(pos)
            h = self.encoder(h, src_key_padding_mask=(attention_mask == 0))

            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            doc_logits = self.doc_head(pooled)
            token_logits = self.token_head(h)

            loss = None
            if doc_labels is not None:
                ce = nn.CrossEntropyLoss()
                loss = ce(doc_logits, doc_labels)
                if token_labels is not None:
                    tce = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss + self.token_loss_weight * tce(
                        token_logits.reshape(-1, N_TOKEN_CLASSES), token_labels.reshape(-1)
                    )
            return {"loss": loss, "doc_logits": doc_logits, "token_logits": token_logits}

    return SmokeForge()
