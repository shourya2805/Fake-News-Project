"""Shared helpers."""

import torch


def get_device():
    """Apple Silicon MPS if available, otherwise CPU. Never CUDA (team is on Macs)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    print(get_device())
