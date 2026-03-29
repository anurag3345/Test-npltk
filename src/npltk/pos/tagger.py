from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from .model import BiLSTMCRF


class POSTagger:
    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "models" / "npltk_pos_tagger.pth")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.word2idx = checkpoint["word2idx"]
        self.char2idx = checkpoint["char2idx"]
        self.tag2idx = checkpoint["tag2idx"]
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        config = checkpoint.get("config", {})

        self.word_pad_idx = self.word2idx["<PAD>"]
        self.word_unk_idx = self.word2idx["<UNK>"]
        self.char_pad_idx = self.char2idx["<PAD>"]
        self.char_unk_idx = self.char2idx["<UNK>"]

        self.model = BiLSTMCRF(
            word_vocab_size=len(self.word2idx),
            char_vocab_size=len(self.char2idx),
            tagset_size=len(self.tag2idx),
            word_emb_dim=config.get("word_emb_dim", 100),
            char_emb_dim=config.get("char_emb_dim", 32),
            char_cnn_out=config.get("char_cnn_out", 64),
            char_kernel_size=config.get("char_kernel_size", 3),
            lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
            lstm_layers=config.get("lstm_layers", 1),
            dropout=config.get("dropout", 0.3),
            word_pad_idx=self.word_pad_idx,
            char_pad_idx=self.char_pad_idx,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _encode_tokens(self, tokens: Sequence[str]):
        word_ids = [self.word2idx.get(tok, self.word_unk_idx) for tok in tokens]
        max_word_len = max((len(tok) for tok in tokens), default=1)

        char_ids = []
        for tok in tokens:
            chars = [self.char2idx.get(ch, self.char_unk_idx) for ch in tok]
            chars += [self.char_pad_idx] * (max_word_len - len(chars))
            char_ids.append(chars)

        words_tensor = torch.tensor([word_ids], dtype=torch.long, device=self.device)
        chars_tensor = torch.tensor([char_ids], dtype=torch.long, device=self.device)
        mask = torch.ones((1, len(tokens)), dtype=torch.bool, device=self.device)

        return words_tensor, chars_tensor, mask

    @torch.no_grad()
    def tag(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []

        words_tensor, chars_tensor, mask = self._encode_tokens(tokens)
        pred_ids = self.model(words_tensor, chars_tensor, mask=mask)[0]
        return [self.idx2tag[i] for i in pred_ids]

    @torch.no_grad()
    def tag_with_tokens(self, tokens: Sequence[str]) -> List[Tuple[str, str]]:
        pred_tags = self.tag(tokens)
        return list(zip(tokens, pred_tags))