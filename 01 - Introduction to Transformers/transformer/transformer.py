import copy

import torch
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
from transformer.training import subsequent_mask


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    # attention layers lears translating sequence to sequence
    attn = MultiHeadedAttention(h, d_model)
    #  self.w_2(self.dropout(self.w_1(x).relu()))
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # a model that adds positional encoding to the input sequence
    position = PositionalEncoding(d_model, dropout)

    # Transformer is encoder-decoder architecture
    model = EncoderDecoder(
        # with 1st layer being encoder that consists of N encoder layers
        # and each encoder and decoder contains 2 sublayers: multi-head attention
        # and feed forward with a dropout
        # EncoderLayer implements a block that has 2 sublsayers:
        # 1. multi-head attention + residual connection
        # 2. feed forward + residual connection
        # dropout is applied in residual connection to f(x) component
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # Decoder also implements self-attention where K V and Q are the same
        # on the input sequence and then another attention layer where K V are from
        # the output of the encoder and Q is from the output of the self-attention layer
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # Embeddings for source and targets
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # ?
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # Uniform distribution scaled by weight
            nn.init.xavier_uniform_(p)
    return model


# feed sequence of encoded tokens to the model
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
