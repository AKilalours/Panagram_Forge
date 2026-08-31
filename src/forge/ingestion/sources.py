"""Phase 1. Source adapters. One class per corpus, all yielding raw dicts.

Each adapter is responsible for recording provenance at the moment of acquisition:
what was fetched, from where, under what license, on what date. Provenance added
later is guesswork.
"""

from __future__ import annotations

from typing import Iterator, Protocol


class Source(Protocol):
    id: str
    license: str

    def stream(self) -> Iterator[dict]:
        """Yield raw upstream records. No cleaning happens here."""
        ...


class FineWebSource:
    """HuggingFaceFW/fineweb, config sample-10BT. Streaming, never fully downloaded."""

    id = "fw"
    license = "ODC-By-1.0"

    def __init__(self, config: str = "sample-10BT", limit: int | None = None) -> None:
        self.config, self.limit = config, limit

    def stream(self) -> Iterator[dict]:
        raise NotImplementedError("Phase 1")


class FineWebEduSource(FineWebSource):
    id = "fwe"


class GutenbergSource:
    id = "gut"
    license = "public-domain-us"

    def stream(self) -> Iterator[dict]:
        raise NotImplementedError("Phase 1")


class GovInfoSource:
    id = "gov"
    license = "public-domain-us-gov"

    def stream(self) -> Iterator[dict]:
        raise NotImplementedError("Phase 1")
