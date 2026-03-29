from typing import cast

import torch
import torch.nn as nn
from torch import ByteTensor, Tensor
from torchcrf import CRF


class CharCNNEncoder(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        char_emb_dim: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.char_emb = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=padding_idx,
        )
        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()

    def forward(self, chars: Tensor) -> Tensor:
        # chars: [B, S, W]
        bsz, seq_len, word_len = chars.size()

        x = chars.view(bsz * seq_len, word_len)   # [B*S, W]
        x = self.char_emb(x)                      # [B*S, W, C]
        x = x.transpose(1, 2)                     # [B*S, C, W]
        x = self.conv(x)                          # [B*S, O, W]
        x = self.relu(x)
        x, _ = torch.max(x, dim=2)                # [B*S, O]
        x = x.view(bsz, seq_len, -1)              # [B, S, O]
        return x


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        tagset_size: int,
        word_emb_dim: int = 100,
        char_emb_dim: int = 32,
        char_cnn_out: int = 64,
        char_kernel_size: int = 3,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        word_pad_idx: int = 0,
        char_pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.word_emb = nn.Embedding(
            num_embeddings=word_vocab_size,
            embedding_dim=word_emb_dim,
            padding_idx=word_pad_idx,
        )

        self.char_encoder = CharCNNEncoder(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            out_channels=char_cnn_out,
            kernel_size=char_kernel_size,
            padding_idx=char_pad_idx,
        )

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=word_emb_dim + char_cnn_out,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(
        self,
        words: Tensor,
        chars: Tensor,
        tags: Tensor | None = None,
        mask: Tensor | None = None,
    ):
        word_repr = self.word_emb(words)
        char_repr = self.char_encoder(chars)

        x = torch.cat([word_repr, char_repr], dim=-1)
        x = self.dropout(x)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        emissions = self.fc(lstm_out)

        crf_mask: ByteTensor
        if mask is None:
            crf_mask = cast(ByteTensor, torch.ones_like(words, dtype=torch.uint8))
        else:
            crf_mask = cast(ByteTensor, mask.to(torch.uint8))

        if tags is not None:
            return -self.crf(emissions, tags, mask=crf_mask, reduction="mean")

        return self.crf.decode(emissions, mask=crf_mask)