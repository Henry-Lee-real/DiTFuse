# DiTFuse
Official implementation of **Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach** (TPAMI 2025)

paper and all detail information will be release before 12.7
Any questions can be consulted -> (Email:lijiayang.cs@gmail.com)

### 📌 TODOs
> - [X] release code  
> - [X] release ckpt
> - [ ] release arxiv
> - [ ] IEEE version paper

## Core Concept:

The core objective of our work is to demonstrate the superiority of a parallel architecture in information control. In our experiments beyond the main paper, I also tried AdaIN-based information injection and T2I-Adapter-style feature-map addition. However, these approaches inevitably cause information from the two modalities to become entangled—numerically mixed together—making it impossible to truly separate the content of the two input images. This is why explicit information disentanglement is necessary, and why a parallel design is the appropriate choice.

In addition, the M3-style synthetic fusion data construction pipeline can significantly improve the performance of the fusion task itself. Finally, with the rapid progress of unified models for visual understanding and generation, we believe fusion tasks should also actively embrace this trend, incorporating strong visual priors into fusion frameworks. We look forward to future advances enabled by such unified architectures.

## 🚀 Overview

[![HuggingFace](https://img.shields.io/badge/HuggingFace-DiTFuse-ffcc4d?logo=huggingface&logoColor=white&style=flat)](https://huggingface.co/lijiayangCS/DiTFuse)

[![Project Page](https://img.shields.io/badge/Project%20Page-DiTFuse-blue?style=flat)](https://ronniejiang.github.io/DiTFuse/)

## Setup

For detailed installation and usage instructions, please refer to ➡️ [`setup.md`](./setup.md).


## Test & Train

### Testing
For testing, please refer to the provided script:➡️ [`test.md`](./test.md)

This script demonstrates how to run DiTFuse in different modes (single, batch, and multi-prompt).


### Training
Training follows the same procedure as OmniGen.


## 📄 Citation

If you use **DiTFuse** in your research, please cite:

```
@article{ditfuse2025,
  title={Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach},
  author={Jiayang Li, Chengjie Jiang, Pengwei Liang, Jiayi Ma, Liqiang Nie, Junjun Jiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```



## ❤️ Acknowledgements

This project is built on **OmniGen**,
a powerful Diffusion Transformer framework developed by VectorSpace Lab.

---

