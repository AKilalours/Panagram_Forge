"""Phase 3. FORGE-Base: one encoder, two heads.

    text -> tokenizer -> transformer encoder -> contextual embeddings
                                                  |            |
                                          document head    token head
                                         (human vs ai)  (human/assisted/generated)

Deliberately not a mixture of experts. Reproducing a production MoE adds engineering
risk without touching the research question, which is about data, not architecture.
"""

from __future__ import annotations


class ForgeBase:
    def __init__(self, backbone: str, max_length: int = 512) -> None:
        raise NotImplementedError("Phase 3")
