# Deploying FORGE to Google Cloud Run

Hugging Face Spaces was the first target and is no longer viable: Docker and Gradio Spaces
on free CPU now require a PRO subscription, and a Static Space cannot run Python. See
`space/PUBLISH.md` if you ever take a PRO subscription; those files still work.

Cloud Run fits because it scales to zero, allows 4 GB of RAM, and its free tier is 180,000
vCPU-seconds, 360,000 GiB-seconds and 2 million requests per month. A demo that gets a few
hundred visits does not come close to that. Google does require a billing card on the
account regardless.

## What is where

    Dockerfile          repository root. Cloud Run's --source build looks for it there.
    .dockerignore       keeps outputs/, data/ and .venv out of the upload
    space/serve.py      reused unchanged; it already honours $PORT, which Cloud Run injects
    space/requirements.txt   the pinned CPU dependency set

The weights are baked into the image at build time from the public weights repo, rather
than fetched at boot. Cold starts are the normal case on a scale-to-zero service, and a
1.5 GB download in front of a reviewer's first click is a worse trade than a large image.

## 1. One-time setup

    gcloud auth login
    gcloud config set project <YOUR_PROJECT_ID>
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
        artifactregistry.googleapis.com

## 2. Deploy

    cd ~/Projects/forge
    gcloud run deploy forge-detect \
        --source . \
        --region us-central1 \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 300 \
        --concurrency 4 \
        --max-instances 3 \
        --port 8080

`--memory 4Gi` is not padding. Each text arm is a DeBERTa-v3-base in float32, roughly
740 MB resident, and both can be loaded at once because the page compares them; the image
detector adds a few hundred MB on top. At 2 GB this container is killed mid-request with
an error a reader cannot interpret.

`--concurrency 4` because scoring is CPU-bound on 2 vCPU. Letting 80 requests in at once,
the Cloud Run default, does not serve them faster, it makes all of them slow.

`--max-instances 3` is a cost guard. Without a ceiling, a crawler or a bad afternoon can
scale this out and spend real money on an account that expected to spend none.

The first build takes roughly ten minutes: it installs CPU torch and downloads 1.5 GB of
checkpoints into the image.

## 3. Check it before sending the link

    SERVICE=$(gcloud run services describe forge-detect --region us-central1 \
        --format 'value(status.url)')
    curl -s "$SERVICE/health" | python -m json.tool

Expect `text_detector_loaded: true`, `image_detector_loaded: true`, and
`polarity_verified: true`. Cloud Run resolves its own Linux wheels, so a dependency that
behaves differently from the local venv shows up here rather than in a stranger's
screenshot.

## 4. Redeploy

Re-run the same `gcloud run deploy` command. It rebuilds from the current working tree.

## Cost guard

Confirm scale-to-zero is really on, which is what keeps the bill at zero between visits:

    gcloud run services describe forge-detect --region us-central1 \
        --format 'value(spec.template.metadata.annotations)'

`autoscaling.knative.dev/minScale` should be absent or `0`. If you ever set a minimum
instance count, the service runs continuously and is billed continuously.
