# Deploying to a Hugging Face Space

Vercel cannot host this. Its serverless bundle limit is about 250 MB unzipped and torch
alone is roughly 800 MB, before the two 735 MB checkpoints. That is not a configuration
problem, it does not fit. Spaces gives a free CPU container with 16 GB of RAM and pulls the
weights from the Hub at runtime instead of bundling them.

## 1. Put the text checkpoints on the Hub

They are too large for git and they are derived artifacts, so `outputs/` stays ignored.

    pip install -U huggingface_hub
    huggingface-cli login          # use a NEW token; the old one is in a chat log

    huggingface-cli repo create forge-detect-weights --type model --private
    huggingface-cli upload AKilalours/forge-detect-weights \
        outputs/forge_min_baseline/best.pt      forge_min_baseline/best.pt
    huggingface-cli upload AKilalours/forge-detect-weights \
        outputs/forge_min_baseline/summary.json forge_min_baseline/summary.json
    huggingface-cli upload AKilalours/forge-detect-weights \
        outputs/forge_min_mirror/best.pt        forge_min_mirror/best.pt
    huggingface-cli upload AKilalours/forge-detect-weights \
        outputs/forge_min_mirror/summary.json   forge_min_mirror/summary.json

## 2. Create the Space

    huggingface-cli repo create forge-detect --type space --space_sdk docker

## 3. Push this repository to it

The Space needs the whole repository, because `api/`, `src/forge/` and
`reports/experiments/` are all read at runtime. The Dockerfile lives at `space/Dockerfile`,
so it is copied to the root of the Space build context.

    git remote add space https://huggingface.co/spaces/AKilalours/forge-detect
    cp space/Dockerfile Dockerfile
    cp space/README.md README-space.md      # keep the repo README; the Space needs its own
    git add Dockerfile && git commit -m "space: docker build"
    git push space master:main

The Space README must be the one with the YAML front matter, so on the Space branch swap it
in. The simplest reliable route is a dedicated `space` branch whose README is
`space/README.md` and whose root Dockerfile is `space/Dockerfile`.

## 4. Set the secrets

In the Space settings, add:

    FORGE_WEIGHTS_REPO = AKilalours/forge-detect-weights
    HF_TOKEN           = a READ token, only if the weights repo is private

Never a write token. The Space only needs to read.

## 5. What to expect

Cold start pulls about 1.5 GB of checkpoints plus the image detector, so the first boot
takes a few minutes. After that a text analysis is a second or two and an image analysis
about three, on two vCPUs.

If the weights repo is unreachable the app still starts: the text tab reports why, and the
image tab, the results tab and the forensics all work. An interface that dies because
weights are missing is worse than one that says so.
