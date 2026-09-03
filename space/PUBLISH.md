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

## 3. Push this repository to the Space

The Space needs the whole repository: `api/`, `src/forge/` and `reports/experiments/` are
all read at runtime. Two files have to sit at the ROOT of the Space, which they do not in
this repo: the Dockerfile, and a README carrying the YAML front matter that tells Spaces
which SDK and port to use.

Do that on a dedicated `space` branch rather than on `master`, so the repository a reviewer
reads keeps its own README and gains no stray root Dockerfile. Re-run these five commands
whenever you want to redeploy; they rebuild the branch from whatever master is:

    git checkout -B space master
    cp space/Dockerfile Dockerfile
    cp space/README.md README.md
    git add -f Dockerfile README.md
    git commit -m "space: root Dockerfile and Space README"

    git remote add space https://huggingface.co/spaces/AKilalours/forge-detect   # once
    git push -f space space:main
    git checkout master

`git push -f` is correct here and only here: the `space` branch is a build artifact rebuilt
from master every time, not history anyone else pulls. Never force-push `master`.

The Docker build context is the repository root, and `space/Dockerfile` already refers to
`space/requirements.txt` and `space/serve.py` by those paths, so copying it to the root
changes nothing about how it builds.

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
image tab and every forensic panel still work. An interface that dies because
weights are missing is worse than one that says so.

## 6. Check it before you send the link

    curl -s https://AKilalours-forge-detect.hf.space/health | python -m json.tool

Expect `text_detector_loaded: true`, `image_detector_loaded: true`, and
`polarity_verified: true`. The Space resolves its own Linux wheels, so a version that
behaves differently from the local venv shows up here rather than in a stranger's
screenshot. If `text_detector_loaded` is false, the message under `text_arms` names the
missing path, which is almost always the weights repo or the token rather than the code.
