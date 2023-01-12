# Transformers United — Stanford CS25 (Fall 21)

This repository contains useful resources for the Stanford CS25 course, including homework solutions and reading notes.

## [01 - Introduction to Transformers](01%20-%20Introduction%20to%20Transformers/README.md)

- Read through [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — a visual guide to the Transformer architecture
- Implement an exercise from [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/). Note that code contains some bugs, so you may need to fix them.
  - `torchtext` had a bug with expired hash for the dataset, so you may need to install the latest version from source (which will require also compile torch, which isn't trivial) or **monkeypatch** it.
  - Walkthrough of the model in the comments of resuling [`transformer`]((01%20-%20Introduction%20to%20Transformers/transformer) library.

## 02 - Language and Human Alignment
