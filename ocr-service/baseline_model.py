# baseline_model.py
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# [1] CNN: 13-layer feature extractor (W/4)
# -----------------------------
class LocalFeatureExtractor(nn.Module):
    """
    13-layer CNN with 5x pooling:
      (2,2), (2,2), (2,1), (2,1), (2,1)
    => Height downsample: 32× | Width downsample: 4×
    Input:  (B,1,H,W)
    Output: (B,512,H/32,W/4)
    """
    def __init__(self, in_ch: int = 1):
        super().__init__()

        def conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.block1 = nn.Sequential(conv(in_ch, 64),  conv(64, 64),  nn.MaxPool2d(2, 2))          # H/2,  W/2
        self.block2 = nn.Sequential(conv(64, 128),    conv(128,128), nn.MaxPool2d(2, 2))          # H/4,  W/4
        self.block3 = nn.Sequential(conv(128,256),    conv(256,256), conv(256,256), nn.MaxPool2d((2,1)))  # H/8,  W/4
        self.block4 = nn.Sequential(conv(256,512),    conv(512,512), conv(512,512), nn.MaxPool2d((2,1)))  # H/16, W/4
        self.block5 = nn.Sequential(conv(512,512),    conv(512,512), conv(512,512), nn.MaxPool2d((2,1)))  # H/32, W/4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)  # (B,  64, H/2,  W/2)
        x = self.block2(x)  # (B, 128, H/4,  W/4)
        x = self.block3(x)  # (B, 256, H/8,  W/4)
        x = self.block4(x)  # (B, 512, H/16, W/4)
        x = self.block5(x)  # (B, 512, H/32, W/4)
        return x


# -----------------------------
# [2] BiLSTM head (sequence modeling + classification)
# -----------------------------
class BiLSTMHead(nn.Module):
    """
    2-layer bidirectional LSTM (hidden=128 per direction) + Linear classifier.

    Input:  x (T, B, input_dim)
    Output: logits (T, B, num_classes)
    """
    def __init__(self, input_dim: int = 320, hidden: int = 128,
                 layers: int = 2, num_classes: int = 96, dropout_lstm: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            bidirectional=True,
            dropout=dropout_lstm,
            batch_first=False,  # expects (T, B, C)
        )
        self.classifier = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)         # (T, B, 2*hidden)
        logits = self.classifier(y) # (T, B, num_classes)
        return logits


# -----------------------------
# [3] Baseline CRNN (CNN -> height-pool -> BiLSTM)
# -----------------------------
@dataclass
class CRNNConfig:
    """
    Baseline CRNN (no Transformer).
    For 128x1024 inputs, the CNN yields (B,512,Hc=4,Wc=256).
    We pool over height to (B,*,1,Wc), then treat Wc as the time dimension T.
    """
    in_ch: int = 1
    feat_dim: int = 320            # feature size fed into the BiLSTM
    use_proj_1x1: bool = True      # 1x1 conv to map 512 -> feat_dim
    pool_type: Literal["avg", "max"] = "avg"
    num_classes: int = 96          # set to VOCAB_SIZE if you include CTC blank at index 0
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    proj_dropout: float = 0.0      # optional dropout after 1x1 conv


class CRNNBaseline(nn.Module):
    """
    CNN -> (height pooling) -> sequence along width -> BiLSTM -> logits
    forward():    returns logits (T, B, num_classes)
    log_probs():  returns log-probs for CTC (T, B, num_classes)
    """
    def __init__(self, cfg: CRNNConfig):
        super().__init__()
        self.cfg = cfg

        # CNN backbone
        self.cnn = LocalFeatureExtractor(cfg.in_ch)

        # Optional 1x1 projection from 512 channels to feat_dim
        if cfg.use_proj_1x1:
            self.proj = nn.Conv2d(512, cfg.feat_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.proj_do = nn.Dropout(cfg.proj_dropout) if cfg.proj_dropout > 0 else nn.Identity()
            lstm_in_dim = cfg.feat_dim
        else:
            self.proj = nn.Identity()
            self.proj_do = nn.Identity()
            lstm_in_dim = 512

        # Pool Hc -> 1 (keep width as time)
        if cfg.pool_type == "avg":
            self.hpool = nn.AdaptiveAvgPool2d((1, None))
        elif cfg.pool_type == "max":
            self.hpool = nn.AdaptiveMaxPool2d((1, None))
        else:
            raise ValueError(f"Unknown pool_type: {cfg.pool_type}")

        # BiLSTM + classifier
        self.head = BiLSTMHead(
            input_dim=lstm_in_dim,
            hidden=cfg.lstm_hidden,
            layers=cfg.lstm_layers,
            num_classes=cfg.num_classes,
            dropout_lstm=cfg.lstm_dropout,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,H,W)
        returns: (T, B, C) where T ~ W/4 (e.g., 256 for W=1024), C = feat_dim or 512
        """
        feat = self.cnn(x)                 # (B,512,Hc,Wc), typically (B,512,4,256)
        feat = self.proj_do(self.proj(feat))  # (B,feat_dim,Hc,Wc) or identity
        feat = self.hpool(feat)            # (B,feat_dim,1,Wc)
        feat = feat.squeeze(2)             # (B,feat_dim,Wc)
        feat = feat.permute(2, 0, 1).contiguous()  # (T=Wc, B, feat_dim)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.forward_features(x)     # (T, B, C)
        logits = self.head(seq)            # (T, B, num_classes)
        return logits

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(x), dim=-1)
