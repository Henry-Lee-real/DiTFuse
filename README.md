# DiTFuse
Official implementation of **Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach** (TPAMI 2026)

Any questions can be consulted -> (Email:lijiayang.cs@gmail.com)

Looking forward to your ⭐！

### 📌 TODOs
> - [X] release code  
> - [X] release ckpt
> - [X] release arxiv
> - [X] IEEE version paper

## Core Concept:

The core objective of our work is to demonstrate the superiority of a parallel architecture in information control. In our experiments beyond the main paper, I also tried AdaIN-based information injection and T2I-Adapter-style feature-map addition. However, these approaches inevitably cause information from the two modalities to become entangled—numerically mixed together—making it impossible to truly separate the content of the two input images. This is why explicit information disentanglement is necessary, and why a parallel design is the appropriate choice.

In addition, the M3-style synthetic fusion data construction pipeline can significantly improve the performance of the fusion task itself. Finally, with the rapid progress of unified models for visual understanding and generation, we believe fusion tasks should also actively embrace this trend, incorporating strong visual priors into fusion frameworks. We look forward to future advances enabled by such unified architectures.

## 🚀 Overview

[![HuggingFace](https://img.shields.io/badge/HuggingFace-DiTFuse-ffcc4d?logo=huggingface&logoColor=white&style=flat)](https://huggingface.co/lijiayangCS/DiTFuse)
[![Project Page](https://img.shields.io/badge/Project%20Page-DiTFuse-blue?style=flat)](https://ronniejiang.github.io/DiTFuse/)
[![arXiv 2512.07170](https://img.shields.io/badge/arXiv-2512.07170-b31b1b?logo=arXiv&logoColor=white&style=flat)](https://arxiv.org/pdf/2512.07170)
[![IEEE TPAMI](https://img.shields.io/badge/IEEE%20TPAMI-Accepted-blue?logo=ieee&logoColor=white)](#)


## Setup

For detailed installation and usage instructions, please refer to ➡️ [`setup.md`](./setup.md).


## Test & Train

### Testing
For testing, please refer to the provided script:➡️ [`test.md`](./test.md)

This script demonstrates how to run DiTFuse in different modes (single, batch, and multi-prompt).

The inference stage requires approximately 12 GB of GPU memory and can be efficiently executed on widely available high-performance GPUs, such as NVIDIA RTX 3090, V100, and RTX 4090.

### Training
Training follows the same procedure as OmniGen.


## 📄 Citation

If you use **DiTFuse** in your research, please cite:

```
@ARTICLE{11297852,
  author={Li, Jiayang and Jiang, Chengjie and Jiang, Junjun and Liang, Pengwei and Ma, Jiayi and Nie, Liqiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach}, 
  year={2026},
  volume={48},
  number={4},
  pages={3970-3987},
  keywords={Semantics;Image fusion;Training;Image segmentation;Transformers;Optimization;Data models;Visual effects;Feature extraction;Electronic mail;Image fusion;DiT;text control},
  doi={10.1109/TPAMI.2025.3642842}}
```



## ❤️ Acknowledgements

This project is built on **OmniGen**,
a powerful Diffusion Transformer framework developed by VectorSpace Lab.

---

