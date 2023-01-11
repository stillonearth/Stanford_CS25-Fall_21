# Transformer Architecture

This is Transformer architecture chart

![image](https://user-images.githubusercontent.com/97428129/211166480-4e8a6fc2-4cc8-48b8-828a-0a38bed4a783.png)

- **BERT** (Bidirectional Encoder Representations from Transformers) is an improvement to Transformer model that introcudes a new block: Bidirectional attention.
- **RoBERTa** (Robustly Optimized BERT Pretraining Approach) is an improvement to BERT that uses a different training objective.
  - Uses Byte-Pair Encoding (BPE) instead of WordPiece tokenization
- **DistilBERT** (Distilled BERT) uses smaller model architecture
  - Can run on a smartphone

## Examples

- [Use untrained transformer to reconstruct input sequences](./01-untrained.ipynb)
- [Train a transformer to reconstruct input sequences](./02-training.ipynb)
- [Train transformer to translate from English to German](./03-text-2-text.ipynb)
