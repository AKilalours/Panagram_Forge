"""What splits and columns does RAID actually publish?

WHY. The out-of-distribution run failed on RAID with "returned no documents": the stream
yielded nothing, so there was nothing to parse. That has several possible causes and they
need different fixes, so this asks the Hub rather than guessing:

    the split name is wrong          -> load_raid passes "extra"; maybe it is not called that
    the config name is required      -> RAID may expose named configs
    the columns changed              -> _require_columns would raise, but only on a non-empty
                                        stream, so a rename plus an empty split looks the same

Prints the available configs and splits with their row counts, then the first row's keys and
a preview. CPU only, no model, about a minute.
"""

from __future__ import annotations

REPO = "liamdugan/raid"


def main() -> int:
    from datasets import get_dataset_config_names, get_dataset_infos

    print(f"\n=== {REPO}")
    try:
        configs = get_dataset_config_names(REPO)
        print("configs:", configs)
    except Exception as error:  # noqa: BLE001
        print("could not list configs:", type(error).__name__, error)
        configs = [None]

    for config in configs or [None]:
        try:
            infos = get_dataset_infos(REPO)
            info = infos.get(config or "default") or next(iter(infos.values()))
            print(f"\nconfig {config!r}")
            for name, split in (info.splits or {}).items():
                rows = getattr(split, "num_examples", "?")
                print(f"   split {name!r}: {rows} rows")
            if info.features:
                print("   columns:", sorted(info.features))
        except Exception as error:  # noqa: BLE001
            print(f"   info failed: {type(error).__name__}: {error}")

    # A single streamed row is the ground truth about column names.
    from forge.evaluation.benchmarks import load_hf

    for split in ("extra", "train", "test"):
        try:
            row = next(iter(load_hf(REPO, None, split)))
        except Exception as error:  # noqa: BLE001
            print(f"\nsplit {split!r}: {type(error).__name__}: {str(error)[:160]}")
            continue
        print(f"\nsplit {split!r} first row keys: {sorted(row)}")
        print(f"   model={row.get('model')!r} domain={row.get('domain')!r} "
              f"attack={row.get('attack')!r}")
        print(f"   generation[:90]={str(row.get('generation'))[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
