# model.py
"""
Minimal model and tokenizer utilities used by agent.py / the notebook.

Students must implement the neural architecture so that checkpoints
trained in the notebook load and decode with the same code here.

Keep the public API stable:

- SPECIAL_TOKENS : Dict[str, str]
- simple_tokenize(s: str) -> List[str]
- encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool=False) -> List[int]
- class Encoder(nn.Module): forward(src, src_lens)
- class Decoder(nn.Module): forward(tgt_in, hidden)
- class Seq2Seq(nn.Module):
    - forward(src, src_lens, tgt_in) -> logits [B,T,V]
    - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Tokenization utilities
# -------------------------

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


def simple_tokenize(s: str) -> List[str]:
    """Lowercase whitespace tokenizer used by both training and inference."""
    return s.strip().lower().split()


def encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool = False) -> List[int]:
    """Map tokens to ids using `stoi`. Optionally wrap with <sos>/<eos>."""
    ids = [stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
    if add_sos_eos:
        ids = [stoi[SPECIAL_TOKENS["sos"]]] + \
            ids + [stoi[SPECIAL_TOKENS["eos"]]]
    return ids


# -------------------------
# Model scaffolding
# -------------------------

class Encoder(nn.Module):
    """
    Student-implemented encoder.
    Expected behavior:
      forward(src: LongTensor[B, S], src_lens: LongTensor[B]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    Returns:
      - outputs: Tensor[B, S, H] (padded time-major outputs)
      - hidden:  RNN-style tuple (h, c) or similar state your decoder expects
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(5000, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, src, src_lens):
        """
        src: [B, S]
        returns encoder_outputs: [B, S, E]
        """
        B, S = src.size()
        pos = torch.arange(S, device=src.device).unsqueeze(0).expand(B, S)
        x = self.emb(src) + self.pos_emb(pos)
        # Padding mask: True = pad, False = real token
        pad_mask = (src == 0)
        enc_out = self.encoder(x, src_key_padding_mask=pad_mask)
        return enc_out, None  # hidden not used by Transformer


class Decoder(nn.Module):
    """
    Student-implemented decoder.
    Expected behavior:
      forward(tgt_in: LongTensor[B, T], hidden) -> Tuple[Tensor, Any]
    Returns:
      - logits: Tensor[B, T, V] (distributions before softmax over target vocab)
      - hidden: updated recurrent state
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(5000, emb_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, tgt_in, encoder_outputs):
        """
        tgt_in: [B, T]
        encoder_outputs: [B, S, E]
        """
        B, T = tgt_in.size()
        pos = torch.arange(T, device=tgt_in.device).unsqueeze(0).expand(B, T)

        x = self.emb(tgt_in) + self.pos_emb(pos)

        # Self-attention mask (causal)
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt_in.device), diagonal=1
        ).bool()

        logits = self.decoder(
            x,
            encoder_outputs,
            tgt_mask=causal_mask,
        )
        return self.proj(logits), None


class Seq2Seq(nn.Module):
    """
    Student-implemented Seq2Seq wrapper that ties Encoder and Decoder.

    Required methods:
      - forward(src, src_lens, tgt_in) -> logits [B, T, V]
      - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
        Greedy decoding should stop at <eos> per sequence and pad remainder with <eos>.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        encoder_outputs, _ = self.encoder(src, src_lens)
        logits, _ = self.decoder(tgt_in, encoder_outputs)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        TODO: implement token-by-token greedy decoding.
        Must return LongTensor[B, max_len]. If <eos> is emitted at step t,
        set positions > t to <eos> for that sequence.
        """
        B = src.size(0)

        encoder_outputs, _ = self.encoder(src, src_lens)

        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for t in range(1, max_len):
            logits, _ = self.decoder(ys, encoder_outputs)
            next_token = logits[:, -1, :].argmax(-1)

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= (next_token == eos_id)

            # If all sequences produced EOS → early exit
            if finished.all():
                break

        # Pad rest with EOS
        out = ys
        if out.size(1) < max_len:
            pad_cols = torch.full(
                (B, max_len - out.size(1)),
                eos_id,
                dtype=torch.long,
                device=src.device,
            )
            out = torch.cat([out, pad_cols], dim=1)

        return out[:, :max_len]
