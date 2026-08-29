"""Smoke test: random tensors of the right shape through FMNVD."""

import torch

from model import FMNVD, SPEECH_TEXT_LENGTH, TEXT_DIM, TITLE_LENGTH
from utils import get_device


def main():
    torch.manual_seed(42)
    device = get_device()
    B = 4

    title = torch.randn(B, TITLE_LENGTH, TEXT_DIM, device=device)
    speech = torch.randn(B, SPEECH_TEXT_LENGTH, TEXT_DIM, device=device)
    title_mask = torch.ones(B, TITLE_LENGTH, device=device)
    speech_mask = torch.ones(B, SPEECH_TEXT_LENGTH, device=device)
    speech_mask[3] = 0            # sample 3 has no transcript
    speech[3] = 0
    has_transcript = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device)

    print(f"device                     : {device}")
    print(f"input title                : {tuple(title.shape)}")
    print(f"input audio_transcript     : {tuple(speech.shape)}")
    print()

    for use_flag in (False, True):
        model = FMNVD(use_has_transcript=use_flag).to(device).eval()
        with torch.no_grad():
            logits = model(title, speech, title_mask, speech_mask, has_transcript)
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tag = "with has_transcript" if use_flag else "without has_transcript"
        print(f"[{tag}]")
        print(f"  output logits            : {tuple(logits.shape)}")
        assert logits.shape == (B, 2), f"expected ({B}, 2), got {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), "non-finite logits"
        print(f"  parameters (total)       : {n_params:,}")
        print(f"  parameters (trainable)   : {n_train:,}")
        print()

    print("PASS: output is (4, 2), all logits finite (empty-transcript sample included)")


if __name__ == "__main__":
    main()
