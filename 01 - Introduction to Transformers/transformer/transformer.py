import copy

import torch.nn as nn

from transformer.layers.multihead_attention import MultiHeadedAttention
from transformer.layers.positionwise_ff import PositionwiseFeedForward
from transformer.layers.positional_encoding import PositionalEncoding
from transformer.layers.embeddings import Embeddings
from transformer.layers.generator import Generator
from transformer.layers.encoder import Encoder
from transformer.layers.encoder_layer import EncoderLayer
from transformer.layers.decoder import Decoder
from transformer.layers.decoder_layer import DecoderLayer
from transformer.layers.encoder_decoder import EncoderDecoder


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model