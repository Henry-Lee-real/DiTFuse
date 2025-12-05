# MaeFuse
Official implementation of **Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach** (TPAMI 2025)

paper and all detail information will be release before 12.7
Any questions can be consulted -> (Email:lijiayang.cs@gmail.com)

### 📌 TODOs
> - [ ] release code  
> - [X] release ckpt
> - [ ] release arxiv
> - [ ] IEEE version paper



## 🚀 Overview

[![HuggingFace](https://img.shields.io/badge/HuggingFace-DiTFuse-ffcc4d?logo=huggingface&logoColor=white&style=flat)](https://huggingface.co/lijiayangCS/DiTFuse)

[![Project Page](https://img.shields.io/badge/🌐–Project%20Page–blue?logo=github&logoColor=white&style=flat)](https://ronniejiang.github.io/DiTFuse/)




## 🧩 Environment Setup

DiTFuse is developed entirely on top of **OmniGen**.
Please install the OmniGen environment **before** running DiTFuse.

👉 **Follow the official OmniGen setup guide:**
[https://github.com/VectorSpaceLab/OmniGen](https://github.com/VectorSpaceLab/OmniGen)

After configuring the OmniGen environment, DiTFuse scripts can be run directly.



## 📦 Model Weights

DiTFuse requires two components:

### **1️⃣ Base Model (Required)**

We use OmniGen-v1 as the foundational diffusion transformer:

👉 **OmniGen-v1:**
[https://huggingface.co/Shitao/OmniGen-v1](https://huggingface.co/Shitao/OmniGen-v1)



### **2️⃣ DiTFuse Fine-tuned Weights (LoRA)**

Our semantic-aware and instruction-controllable LoRA modules:

👉 **DiTFuse LoRA Weights:**
[https://huggingface.co/lijiayangCS/DiTFuse](https://huggingface.co/lijiayangCS/DiTFuse)

These LoRA weights must be merged into the OmniGen base model before inference.



## 📁 Project Structure (Preview)

```
DiTFuse/
│── scripts/
│   ├── run_single.py
│   ├── run_batch.py
│   ├── run_prompt.py
│── configs/
│── checkpoints/
│── README.md
```


## ▶️ Quick Start (Example)

### **Single Pair Fusion**

```bash
python run.py \
  --mode single \
  --image1 path/to/img1.png \
  --image2 path/to/img2.png \
  --prompt_single "Fuse the two images while preserving thermal targets."
```

### **Batch Fusion**

```bash
python run.py \
  --mode batch \
  --image_dir ./data/fusion_pairs \
  --prompt_batch "High-clarity multispectral fusion."
```

### **Prompt Library Fusion**

```bash
python run.py \
  --mode prompt \
  --prompt_file prompts.txt \
  --image1 img_a.png \
  --image2 img_b.png
```



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

This project is built on top of **OmniGen**,
a powerful Diffusion Transformer framework developed by VectorSpace Lab.

---

