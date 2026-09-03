"""Generate demo texts from the families the detector was actually trained on. CPU, local.

WHY THIS SCRIPT EXISTS. Pasting ChatGPT output into the text tab returns "no AI detected",
and that is not a bug: the arms were trained on Qwen, Phi, SmolLM and Falcon at 1.7B to 3.8B
parameters, and the out-of-distribution evaluation measured a 63% to 96% miss rate on unseen
generators. GPT-4-class text is an unseen generator. A demo that invites someone to paste
ChatGPT output is a demo of the one weakness the project already measured and published.

So the demo needs AI text from a HELD-IN family. SmolLM2-360M-Instruct is one of them, it is
about 700 MB, and it generates a few hundred words on a laptop CPU in under a minute. No GPU,
no API key, no pod.

WHAT TO USE FOR THE HUMAN SIDE. Your own writing. Not public-domain literature, which is a
different distribution from the FineWeb web prose the arms trained on and would test the
wrong thing. An email you sent, a paragraph of your notes, a section of your own writeup:
that is genuinely human, genuinely in distribution, and it exercises the part of this
detector that is actually strong, a 0.05% false-positive rate.

    python scripts/make_demo_texts.py --out demo/

Writes one file per sample plus a manifest naming the generator and the expected verdict.
The expected verdict is what the committed evaluation predicts, not what the detector says:
a sample the detector then misses is a finding to show, not a file to delete.
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Held-in families, from configs/generation/generators_minimal.yaml. Smallest first: this
# runs on a laptop, and the point is in-distribution text rather than a big model.
HELD_IN = {
    "smollm": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
}

PROMPTS = [
    "Write four paragraphs explaining how public libraries changed between 1900 and 1970.",
    "Write four paragraphs about why bridges are inspected and what inspectors look for.",
    "Write four paragraphs describing how a city decides where to plant street trees.",
    "Write four paragraphs on the history of standardised shipping containers.",
    "Write four paragraphs about how weather forecasts are produced and why they fail.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demo", help="directory to write samples into")
    parser.add_argument("--family", default="smollm", choices=sorted(HELD_IN))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--words", type=int, default=260)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = HELD_IN[args.family]
    print(f"loading {model_id} on CPU, this downloads once")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.float()          # CPU inference; a half checkpoint fails at the first matmul
    model.eval()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for index, prompt in enumerate(PROMPTS[: args.n], start=1):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(args.words * 1.4),
                do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        body = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        path = out / f"ai_{args.family}_{index:02d}.txt"
        path.write_text(body + "\n")
        manifest.append({
            "file": path.name,
            "label": "ai",
            "generator": model_id,
            "family": args.family,
            "held_in": True,
            "words": len(body.split()),
            "expected_verdict": "ai",
            "why": (
                "A held-in generator family. The committed in-distribution evaluation puts "
                "FNR at 0.43% for the mirror arm, so this should be caught."
            ),
        })
        print(f"  {path.name}  {len(body.split())} words")

    (out / "MANIFEST.json").write_text(json.dumps({
        "samples": manifest,
        "human_side": (
            "Use your own writing for the human samples: an email, your notes, a section of "
            "your writeup. Public-domain literature is a different distribution from the "
            "FineWeb web prose these arms trained on and would test the wrong thing."
        ),
        "expected_verdicts_are_predictions": (
            "Taken from the committed evaluation, not from running the detector. A sample it "
            "then misses is a finding worth showing, not a file to delete."
        ),
        "not_in_distribution": (
            "ChatGPT, Claude and Gemini are unseen generators for these checkpoints. The "
            "measured miss rate on unseen generators is 63% to 96%, so those will mostly "
            "read as no AI detected. That is the published result, not a malfunction."
        ),
    }, indent=2) + "\n")
    print(f"\nwritten {out}/MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
