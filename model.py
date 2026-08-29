"""FMNVD — title + audio path only (paper's "w/o Frames" ablation).

Dimensions are taken verbatim from the authors' `FMNVD.py`:

    text_dim = 768, title_length = 32, speech_text_length = 64,
    dim = 128, dropout = 0.1, num_heads = 4

Forward pass:

    title  (B, 32, 768) --linear_title--> (B, 32, 128)
    speech (B, 64, 768) --linear_speech-> (B, 64, 128)
              |
              +-- co_attention_ts (bidirectional, title <-> speech)
              |
    mean over token dim -> (B, 128) each
    unsqueeze + concat  -> (B, 2, 128)
    TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
    mean over sequence  -> (B, 128)
    Linear(128, 2)      -> logits (B, 2)

The original concatenates three modalities into (B, 3, 128); we drop the visual
one, so ours is (B, 2, 128). Intended consequence of the ablation scope.
"""

import torch
import torch.nn as nn

from modules.coattention import co_attention

TEXT_DIM = 768
TITLE_LENGTH = 32
SPEECH_TEXT_LENGTH = 64
DIM = 128
DROPOUT = 0.1
NUM_HEADS = 4


class FMNVD(nn.Module):
    def __init__(self, text_dim=TEXT_DIM, dim=DIM, dropout=DROPOUT,
                 num_heads=NUM_HEADS, title_length=TITLE_LENGTH,
                 speech_text_length=SPEECH_TEXT_LENGTH, num_classes=2,
                 use_has_transcript=False):
        super().__init__()
        self.use_has_transcript = use_has_transcript

        self.linear_title = nn.Sequential(
            nn.Linear(text_dim, dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.linear_speech = nn.Sequential(
            nn.Linear(text_dim, dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.co_attention_ts = co_attention(
            d_k=dim // num_heads, d_v=dim // num_heads, n_heads=num_heads,
            dropout=dropout, d_model=dim,
            visual_len=speech_text_length, sen_len=title_length,
            fea_v=dim, fea_s=dim, pos=False,
        )

        self.trm = nn.TransformerEncoderLayer(
            d_model=dim, nhead=2, batch_first=True
        )

        clf_in = dim + 1 if use_has_transcript else dim
        self.classifier = nn.Linear(clf_in, num_classes)

    def forward(self, title, speech, title_mask=None, speech_mask=None,
                has_transcript=None):
        """title: (B, 32, 768), speech: (B, 64, 768) -> logits (B, 2)."""
        fea_title = self.linear_title(title)     # (B, 32, 128)
        fea_speech = self.linear_speech(speech)  # (B, 64, 128)

        # v = speech/audio stream, s = title stream (authors' naming).
        fea_speech, fea_title = self.co_attention_ts(
            v=fea_speech, s=fea_title, v_len=speech_mask, s_len=title_mask
        )

        fea_speech = self._masked_mean(fea_speech, speech_mask)  # (B, 128)
        fea_title = self._masked_mean(fea_title, title_mask)     # (B, 128)

        fea = torch.cat(
            [fea_title.unsqueeze(1), fea_speech.unsqueeze(1)], dim=1
        )                                                        # (B, 2, 128)
        fea = self.trm(fea)
        fea = torch.mean(fea, dim=1)                             # (B, 128)

        if self.use_has_transcript:
            if has_transcript is None:
                raise ValueError(
                    "use_has_transcript=True but has_transcript was not passed"
                )
            flag = has_transcript.to(fea.dtype).view(-1, 1)
            fea = torch.cat([fea, flag], dim=1)                  # (B, 129)

        return self.classifier(fea)                              # (B, 2)

    @staticmethod
    def _masked_mean(x, mask):
        """Mean over the token dim, ignoring padded positions when a mask is given."""
        if mask is None or not torch.is_tensor(mask) or mask.dim() != 2:
            return torch.mean(x, dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)               # (B, L, 1)
        denom = m.sum(dim=1).clamp(min=1.0)              # empty stream -> zeros
        return (x * m).sum(dim=1) / denom
