# reseeta_model.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
                nn.ReLU(inplace=True)
            )
        self.block1 = nn.Sequential(conv(in_ch, 64),  conv(64, 64),  nn.MaxPool2d(2,2))         # H/2,  W/2
        self.block2 = nn.Sequential(conv(64, 128),    conv(128,128), nn.MaxPool2d(2,2))         # H/4,  W/4
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
# [2] ViT encoder-only over CNN features (patchify along width)
# -----------------------------
class ViTEncoderForCRNN(nn.Module):
    """
    ViT encoder-only for CRNN:
      - Patch Embedding over CNN feature map (full height x patch_w along width)
      - Learned Positional Embedding
      - Transformer Encoder: depth=3, nhead=4, GELU MLPs, residuals (norm_first=True)
      - Final LayerNorm + Dropout(0.5)
    Output: (T, B, embed_dim), ready for BiLSTM where T = number of patches along width.

    With 128x1024 inputs and the CNN above → (B,512,4,256).
    T depends on patch_w: patch_w=1 -> T=256; patch_w=2 -> T=128; patch_w=4 -> T=64.
    """
    def __init__(
        self,
        in_ch: int = 512,
        embed_dim: int = 320,
        nhead: int = 4,
        depth: int = 3,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.1,
        final_dropout: float = 0.5,
        max_len: int = 4096,
        patch_w: int = 1,
        norm_first: bool = True,
    ):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.patch_w = patch_w

        # Built lazily after seeing Hc
        self.proj: Optional[nn.Linear] = None

        self.pos_embed = nn.Parameter(torch.zeros(max_len, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=resid_dropout,
            activation="gelu",
            batch_first=False,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_dropout = nn.Dropout(final_dropout)

    def _lazy_build_proj(self, Hc: int, device: torch.device):
        flat_dim = self.in_ch * Hc * self.patch_w
        if (self.proj is None) or (self.proj.in_features != flat_dim):
            self.proj = nn.Linear(flat_dim, self.embed_dim, bias=False).to(device)

    @staticmethod
    def _check_divisible(x: int, by: int, name="width"):
        if x % by != 0:
            raise ValueError(f"{name}={x} must be divisible by patch_w={by}")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C=512, Hc, Wc) from CNN
        returns: (T, B, embed_dim)
        """
        B, C, Hc, Wc = feat.shape
        assert C == self.in_ch, f"Expected C={self.in_ch}, got {C}"
        self._check_divisible(Wc, self.patch_w, name="Wc")
        self._lazy_build_proj(Hc, feat.device)

        # Patchify over width: kernel (Hc, patch_w), stride (Hc, patch_w)
        patches = F.unfold(
            feat, kernel_size=(Hc, self.patch_w), stride=(Hc, self.patch_w), padding=0
        )  # (B, C*Hc*patch_w, L)
        L = patches.shape[-1]             # tokens along width
        tokens = patches.transpose(1, 2)  # (B, L, C*Hc*patch_w)
        tokens = self.proj(tokens)        # (B, L, embed_dim)

        x = tokens.transpose(0, 1)        # (T=L, B, embed_dim)

        if L > self.pos_embed.shape[0]:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.pos_embed.shape[0]}")
        x = x + self.pos_embed[:L]        # learned positional embedding

        x = self.encoder(x)               # (T, B, embed_dim)
        x = self.final_norm(x)
        x = self.final_dropout(x)
        return x


# -----------------------------
# [3] BiLSTM head for sequence modeling + character classification
# -----------------------------
class BiLSTMHead(nn.Module):
    """
    [6] Bi-LSTM (Sequence Modeling)
         - 2-layer bidirectional LSTM, 128 hidden per direction
    [7] Linear (Character Classification)
         - Output: (T, B, num_classes)
    """
    def __init__(self, input_dim: int = 320, hidden: int = 128,
                 layers: int = 2, num_classes: int = 95, dropout_lstm: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            bidirectional=True,
            dropout=dropout_lstm,
            batch_first=False  # expects (T, B, C)
        )
        self.classifier = nn.Linear(2 * hidden, num_classes)  # 2x for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, input_dim)   # tokens from ViT encoder
        returns:
            logits: (T, B, num_classes)  # CTC-ready (no softmax in forward)
        """
        y, _ = self.rnn(x)                 # (T, B, 2*hidden)
        logits = self.classifier(y)        # (T, B, num_classes)
        return logits

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities for CTC (still (T,B,C))."""
        return F.log_softmax(self.forward(x), dim=-1)


# -----------------------------
# [4] Wrapper model: CNN -> ViT -> BiLSTM
# -----------------------------
@dataclass
class ViTCRNNConfig:
    in_ch: int = 1
    embed_dim: int = 320
    nhead: int = 4
    depth: int = 3
    patch_w: int = 1          # you chose 1 for max detail (T=256 on 1024-wide)
    num_classes: int = 95     # matches charset_base length; add blank in training config if needed
    final_dropout: float = 0.5
    resid_dropout: float = 0.1
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    norm_first: bool = True   # set False to silence nested-tensor warning

class ViTCRNN(nn.Module):
    """
    End-to-end forward:
      images (B,1,H,W)
        -> CNN (B,512,H/32,W/4)
        -> ViTEncoder (T,W/patch_w tokens): (T,B,embed_dim)
        -> BiLSTMHead -> logits (T,B,num_classes)
    """
    def __init__(self, cfg: ViTCRNNConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = LocalFeatureExtractor(cfg.in_ch)
        self.vit = ViTEncoderForCRNN(
            in_ch=512,
            embed_dim=cfg.embed_dim,
            nhead=cfg.nhead,
            depth=cfg.depth,
            resid_dropout=cfg.resid_dropout,
            final_dropout=cfg.final_dropout,
            patch_w=cfg.patch_w,
            norm_first=cfg.norm_first,
        )
        self.head = BiLSTMHead(
            input_dim=cfg.embed_dim,
            hidden=cfg.lstm_hidden,
            layers=cfg.lstm_layers,
            num_classes=cfg.num_classes,
            dropout_lstm=cfg.lstm_dropout,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ViT tokens (T,B,embed_dim)."""
        feat = self.cnn(x)         # (B,512,Hc,Wc) -> typically (B,512,4,256) for 128x1024
        tokens = self.vit(feat)    # (T,B,embed_dim)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-timestep logits (T,B,num_classes)."""
        return self.head(self.forward_features(x))

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-timestep log-probabilities (T,B,num_classes) for CTC."""
        return self.head.log_probs(self.forward_features(x))
