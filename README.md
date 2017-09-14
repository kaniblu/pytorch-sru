# PyTorch SRU

This is just a independently packaged and properly interfaced SRU in PyTorch.
**The credits for main source code all belong @taolei87 (https://github.com/taolei87/sru)**.

This main difference between this package and the author's source code is that

  * **Basic handling for `PackedSequence` inputs**. However, if there are enough demands
    for it, further optimization code can be implemented to leverage packed
    data structures in CUDA-level.

  * **Handling of variable length sequences in a mini-batch**. The capability to
    handle `PackedSequence` also means that the underlying CUDA-level code must
    support variable sequence lengths in a mini-batch. Some basic codes
    have been modified to output only the last hidden state of each sequence (of
    variable lengths).

We plan to update this package as soon as the author puts out additional
functionalities (layer normalization etc.)