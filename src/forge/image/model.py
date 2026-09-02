"""FORGE-Image: a frozen visual backbone with two heads.

SHAPE

    image -> backbone -> [CLS token, patch tokens]
                            |            |
                            v            v
                       global head   local head
                            |            |
                     is it generated   where is the evidence

THE DECISIONS THAT ARE NOT OBVIOUS

1. THE LOSS IS COMPUTED AT PATCH RESOLUTION, NOT PIXEL RESOLUTION. The local head emits one
   logit per patch, a 16x16 grid for a 224-pixel image at patch size 14. The obvious move is
   to upsample those logits to the mask's resolution and take a pixel-wise loss. That makes
   the loss dominated by interpolation: the model is graded on how a resize behaved rather
   than on what it predicted. Downsampling the MASK to the patch grid instead grades exactly
   what the head can express. Upsampling belongs at display time, not in the objective.

2. A WHOLE-IMAGE SAMPLE STILL SUPERVISES THE LOCAL HEAD. A fully generated image has an
   all-ones mask, a photograph an all-zeros mask. Treating those as "no localisation label"
   would train the local head only on composites, which are a small and artificial slice, and
   the head would then behave strangely on ordinary inputs.

3. THE GLOBAL HEAD READS BOTH THE CLS TOKEN AND POOLED PATCH TOKENS. CLS alone tends to
   summarise the scene, which is what we are explicitly trying not to classify on. Pooled
   patch evidence keeps the global decision tied to the same signal the local head uses.

4. THREE VERDICTS, NOT TWO. Same rule as the text detector: a model forced to answer will
   guess on an image it has no business judging. Abstention is part of the output.

The backbone is injectable so the architecture, the loss and the decision rule are all
testable on CPU without downloading DINOv3.
"""

from __future__ import annotations

from dataclasses import dataclass

BACKBONE_DEFAULT = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@dataclass(frozen=True)
class ForgeImageConfig:
    backbone: str = BACKBONE_DEFAULT
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    local_loss_weight: float = 0.5
    freeze_backbone: bool = True
    # Below this, the model says "uncertain" rather than guessing. Calibrated on val, not
    # chosen here; this is only the default.
    abstain_below: float = 0.65

    @property
    def grid(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} is not divisible by patch_size "
                f"{self.patch_size}; the patch grid would not tile the image"
            )
        return self.image_size // self.patch_size


def downsample_mask(mask, grid: int):
    """Mask at pixel resolution -> mask at patch resolution, as a fraction per patch.

    Average pooling rather than nearest-neighbour: a patch that is half covered should read
    as half covered, not as arbitrarily on or off. The local head is then asked to predict a
    soft target, which is what the ground truth actually is at that resolution.
    """
    import torch.nn.functional as functional

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return functional.adaptive_avg_pool2d(mask.float(), (grid, grid)).squeeze(1)


def build_model(config: ForgeImageConfig, backbone=None):
    """Assemble the detector. `backbone` is injectable so tests need no download."""
    import torch
    from torch import nn

    class ForgeImage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.backbone = backbone if backbone is not None else _load_backbone(config)
            if config.freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            self.local_head = nn.Linear(config.hidden_size, 1)
            # CLS and pooled patch evidence, concatenated. See decision 3.
            self.global_head = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1),
            )

        def forward(self, pixel_values, mask=None, label=None):
            cls_token, patch_tokens = self.backbone(pixel_values)

            patch_logits = self.local_head(patch_tokens).squeeze(-1)
            grid = self.config.grid
            patch_map = patch_logits.view(-1, grid, grid)

            pooled = patch_tokens.mean(dim=1)
            doc_logit = self.global_head(torch.cat([cls_token, pooled], dim=-1)).squeeze(-1)

            out = {"logit": doc_logit, "patch_logits": patch_map}
            if label is None and mask is None:
                return out

            loss = torch.zeros((), device=doc_logit.device)
            bce = nn.functional.binary_cross_entropy_with_logits
            if label is not None:
                out["global_loss"] = bce(doc_logit, label.float())
                loss = loss + out["global_loss"]
            if mask is not None:
                target = downsample_mask(mask, grid)
                out["local_loss"] = bce(patch_map, target)
                loss = loss + self.config.local_loss_weight * out["local_loss"]
            out["loss"] = loss
            return out

    return ForgeImage()


def _load_backbone(config: ForgeImageConfig):  # pragma: no cover - needs a download
    """Wrap a HuggingFace vision model into the (cls, patches) contract used above."""
    import torch
    from transformers import AutoModel

    inner = AutoModel.from_pretrained(config.backbone)

    class Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            states = self.inner(pixel_values=pixel_values).last_hidden_state
            # Some checkpoints prepend register tokens after CLS; keep the LAST grid*grid
            # tokens so the patch map always tiles the image regardless of how many extra
            # tokens the checkpoint carries.
            wanted = config.grid * config.grid
            return states[:, 0], states[:, -wanted:]

    return Wrapped()


def verdict(probability: float, abstain_below: float) -> str:
    """human, ai, or uncertain.

    The band is symmetric around 0.5: a model that is unsure in either direction should say
    so rather than defaulting to the majority class, which is what an asymmetric rule
    quietly does.
    """
    if not 0.5 < abstain_below <= 1.0:
        raise ValueError(f"abstain_below must be in (0.5, 1.0], got {abstain_below}")
    if probability >= abstain_below:
        return "ai"
    if probability <= 1.0 - abstain_below:
        return "human"
    return "uncertain"


def heatmap(patch_logits, size: int):
    """Patch logits -> a pixel-resolution probability map, for display only.

    Upsampling happens HERE and never in the loss. See decision 1.
    """
    import torch
    import torch.nn.functional as functional

    probabilities = torch.sigmoid(patch_logits)
    if probabilities.dim() == 2:
        probabilities = probabilities.unsqueeze(0)
    return functional.interpolate(
        probabilities.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(1)
