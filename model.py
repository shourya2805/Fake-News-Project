"""FMNVD, in two modes selected by `modalities`.

Dimensions are taken verbatim from the authors' `FMNVD.py`:

    text_dim = 768, title_length = 32, speech_text_length = 64,
    clip_dim = 512, motion_dim = 2048, num_frames = 55,
    dim = 128, dropout = 0.1, num_heads = 4, d_k = d_v = 128

d_k defaults to dim // num_heads = 32 (the Phase 1 value); pass d_k=128 for the
authors' setting.

modalities="text" (paper's "w/o Frames" ablation):

    title  (B, 32, 768) --linear_title--> (B, 32, 128)
    speech (B, 64, 768) --linear_speech-> (B, 64, 128)
    co_attention_ts (title <-> speech)
    mean over tokens, concat (title, speech) -> (B, 2, 128)
    TransformerEncoderLayer -> mean -> Linear(128, 2)

modalities="full" (paper Fig. 5):

    clip   (B, 55, 512)  --linear_clip---> (B, 55, 128)
    motion (B, 55, 2048) --linear_motion-> (B, 55, 128)
    per-step gate  a = sigmoid(W[motion; clip] + b)            (B, 55, 1)
    visual = a * motion + (1 - a) * clip                       (B, 55, 128)
    co_attention_tv (enhanced title <-> visual), 55-step sequence
    mean over tokens / steps, concat (title, speech, visual) -> (B, 3, 128)
    TransformerEncoderLayer -> mean -> Linear(128, 2)
"""

import torch
import torch.nn as nn

from modules.coattention import co_attention

TEXT_DIM = 768
TITLE_LENGTH = 32
SPEECH_TEXT_LENGTH = 64
CLIP_DIM = 512
MOTION_DIM = 2048
NUM_FRAMES = 55
DIM = 128
DROPOUT = 0.1
NUM_HEADS = 4
MODALITIES = ("text", "full")


class FMNVD(nn.Module):
    def __init__(self, text_dim=TEXT_DIM, dim=DIM, dropout=DROPOUT,
                 num_heads=NUM_HEADS, title_length=TITLE_LENGTH,
                 speech_text_length=SPEECH_TEXT_LENGTH, num_classes=2,
                 use_has_transcript=False, modalities="text",
                 clip_dim=CLIP_DIM, motion_dim=MOTION_DIM, num_frames=NUM_FRAMES,
                 d_k=None):
        super().__init__()
        if modalities not in MODALITIES:
            raise ValueError(f"modalities must be one of {MODALITIES}, got {modalities!r}")
        self.use_has_transcript = use_has_transcript
        self.modalities = modalities
        d_k = d_k or dim // num_heads
        self.d_k = d_k

        self.linear_title = nn.Sequential(
            nn.Linear(text_dim, dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.linear_speech = nn.Sequential(
            nn.Linear(text_dim, dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.co_attention_ts = co_attention(
            d_k=d_k, d_v=d_k, n_heads=num_heads,
            dropout=dropout, d_model=dim,
            visual_len=speech_text_length, sen_len=title_length,
            fea_v=dim, fea_s=dim, pos=False,
        )

        self.trm = nn.TransformerEncoderLayer(
            d_model=dim, nhead=2, batch_first=True
        )

        clf_in = dim + 1 if use_has_transcript else dim
        self.classifier = nn.Linear(clf_in, num_classes)

        if modalities == "full":
            self.linear_motion = nn.Sequential(
                nn.Linear(motion_dim, dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.linear_clip = nn.Sequential(
                nn.Linear(clip_dim, dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.linear_attn = nn.Sequential(
                nn.Linear(dim * 2, 1), nn.Sigmoid()
            )
            self.co_attention_tv = co_attention(
                d_k=d_k, d_v=d_k, n_heads=num_heads,
                dropout=dropout, d_model=dim,
                visual_len=num_frames, sen_len=title_length,
                fea_v=dim, fea_s=dim, pos=False,
            )

    def forward(self, title, speech, title_mask=None, speech_mask=None,
                has_transcript=None, clip=None, motion=None):
        fea_title = self.linear_title(title)
        fea_speech = self.linear_speech(speech)

        fea_speech, fea_title = self.co_attention_ts(
            v=fea_speech, s=fea_title, v_len=speech_mask, s_len=title_mask
        )

        streams = []
        if self.modalities == "full":
            if clip is None or motion is None:
                raise ValueError("modalities='full' requires clip and motion inputs")
            fea_visual = self.gated_visual(clip, motion)
            fea_visual, fea_title = self.co_attention_tv(
                v=fea_visual, s=fea_title, v_len=None, s_len=title_mask
            )
            streams.append(torch.mean(fea_visual, dim=1))

        fea_speech = self._masked_mean(fea_speech, speech_mask)
        fea_title = self._masked_mean(fea_title, title_mask)

        fea = torch.stack([fea_title, fea_speech] + streams, dim=1)
        fea = self.trm(fea)
        fea = torch.mean(fea, dim=1)

        if self.use_has_transcript:
            if has_transcript is None:
                raise ValueError(
                    "use_has_transcript=True but has_transcript was not passed"
                )
            flag = has_transcript.to(fea.dtype).view(-1, 1)
            fea = torch.cat([fea, flag], dim=1)

        return self.classifier(fea)

    def gated_visual(self, clip, motion):
        fea_motion = self.linear_motion(motion)
        fea_clip = self.linear_clip(clip)
        gate = self.linear_attn(torch.cat([fea_motion, fea_clip], dim=-1))
        return gate * fea_motion + (1 - gate) * fea_clip

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None or not torch.is_tensor(mask) or mask.dim() != 2:
            return torch.mean(x, dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom
