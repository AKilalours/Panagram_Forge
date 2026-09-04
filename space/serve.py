"""Space entrypoint: fetch the text checkpoints, then serve.

The two text checkpoints are about 735 MB each and do not belong in git. They live in a
Hugging Face model repository and are pulled once into `outputs/` on cold start, which is
where `forge.inference.scorer` already looks.

If the download fails, or FORGE_WEIGHTS_REPO is unset, the app still starts and the text tab
reports that no arm could be loaded, naming the reason. A detection interface that dies
because weights are missing is worse than one that says so: the image tab, the results tab
and the forensics all work without them.
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO = os.getenv("FORGE_WEIGHTS_REPO", "").strip()
ARMS = ("forge_min_baseline", "forge_min_mirror")
WANTED = ("best.pt", "summary.json")
OUT = pathlib.Path("outputs")


def _already_present() -> bool:
    return all((OUT / arm / name).exists() for arm in ARMS for name in WANTED)


def fetch() -> None:
    # The Cloud Run image bakes the checkpoints in at build time, so on that deployment
    # there is nothing to fetch and FORGE_WEIGHTS_REPO is deliberately unset. Saying
    # "starting without text checkpoints" there would be false, and it is the exact
    # sentence someone would read while debugging a text tab that in fact works.
    if _already_present():
        print("checkpoints already present; no download needed", flush=True)
        return
    if not REPO:
        print("FORGE_WEIGHTS_REPO is not set and no local checkpoints found; "
              "the text tab will report why", flush=True)
        return
    from huggingface_hub import hf_hub_download

    for arm in ARMS:
        for name in WANTED:
            target = OUT / arm / name
            if target.exists():
                continue
            try:
                path = hf_hub_download(
                    repo_id=REPO,
                    filename=f"{arm}/{name}",
                    token=os.getenv("HF_TOKEN") or None,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(pathlib.Path(path).read_bytes())
                print(f"fetched {arm}/{name}", flush=True)
            except Exception as error:  # noqa: BLE001 - reported, never fatal
                print(f"could not fetch {arm}/{name}: {type(error).__name__}: {error}",
                      flush=True)


def main() -> int:
    fetch()
    import uvicorn

    uvicorn.run(
        "api.forge_app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,          # the models are loaded per process; a second worker doubles RAM
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
