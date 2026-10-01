"""Smoke test: random tensors of the right shape through FMNVD, both modalities."""

import torch

from model import (CLIP_DIM, FMNVD, MOTION_DIM, NUM_FRAMES, SPEECH_TEXT_LENGTH,
                   TEXT_DIM, TITLE_LENGTH)
from utils import get_device


def main():
    torch.manual_seed(42)
    device = get_device()
    B = 4

    title = torch.randn(B, TITLE_LENGTH, TEXT_DIM, device=device)
    speech = torch.randn(B, SPEECH_TEXT_LENGTH, TEXT_DIM, device=device)
    title_mask = torch.ones(B, TITLE_LENGTH, device=device)
    speech_mask = torch.ones(B, SPEECH_TEXT_LENGTH, device=device)
    speech_mask[3] = 0
    speech[3] = 0
    has_transcript = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device)
    clip = torch.randn(B, NUM_FRAMES, CLIP_DIM, device=device)
    motion = torch.relu(torch.randn(B, NUM_FRAMES, MOTION_DIM, device=device))

    print(f"device                     : {device}")
    print(f"input title                : {tuple(title.shape)}")
    print(f"input audio_transcript     : {tuple(speech.shape)}")
    print(f"input clip (full only)     : {tuple(clip.shape)}")
    print(f"input motion (full only)   : {tuple(motion.shape)}")
    print()

    for modalities in ("text", "full"):
        for use_flag in (False, True):
            model = FMNVD(use_has_transcript=use_flag, modalities=modalities).to(device).eval()
            with torch.no_grad():
                logits = model(title, speech, title_mask, speech_mask, has_transcript,
                               clip=clip, motion=motion)
            n_params = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            tag = "with has_transcript" if use_flag else "without has_transcript"
            print(f"[{modalities} | {tag}]")
            print(f"  output logits            : {tuple(logits.shape)}")
            assert logits.shape == (B, 2), f"expected ({B}, 2), got {tuple(logits.shape)}"
            assert torch.isfinite(logits).all(), "non-finite logits"
            print(f"  parameters (total)       : {n_params:,}")
            print(f"  parameters (trainable)   : {n_train:,}")
            print()

    full = FMNVD(modalities="full").to(device).eval()
    with torch.no_grad():
        gated = full.gated_visual(clip, motion)
        logits = full(title, speech, title_mask, speech_mask, clip=clip, motion=motion)
        logits_swapped = full(title, speech, title_mask, speech_mask, clip=clip, motion=motion.flip(0))
    print(f"[full | internals]")
    print(f"  gated visual sequence    : {tuple(gated.shape)}")
    assert gated.shape == (B, NUM_FRAMES, full.linear_clip[0].out_features)
    print(f"  logits depend on video   : {not torch.allclose(logits, logits_swapped)}")
    assert not torch.allclose(logits, logits_swapped), "video input has no effect"
    try:
        full(title, speech, title_mask, speech_mask)
        raise AssertionError("full model accepted missing video inputs")
    except ValueError:
        print(f"  missing video rejected   : True")
    print()

    print("PASS: output is (4, 2) in both modes, all logits finite (empty-transcript sample included)")


if __name__ == "__main__":
    main()
