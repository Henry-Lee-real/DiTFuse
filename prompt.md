# Prompt Design for DiTFuse (Training & Inference)

This document describes the **instruction prompt design** used in the paper:

**“Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach (DiTFuse)”**. 

---

## 1. Prompt Structure 

Each prompt consists of **four components**:

1. **[TASK] Tag**
2. **\<SUBTASK> Tag (Optional)**
3. **Two image placeholders**
4. **Free-form natural language instruction**

### 1.1 General Template

```text
[TASK]. <SUBTASK>. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate the image based on the following requirements: XXX.
```

---

## 2. Task Tags

| Tag | Description |
|------|------------|
| `[FUSION]` | All classical fusion tasks and M3 training |
| `[CONTROL]` | Text-controlled image-level adjustments |
| `[SEG]` | Instruction-following segmentation |

---

## 3. Subtask Tags

### 3.1 Fusion Subtasks

| Subtag | Meaning |
|--------|--------|
| `<MULTI-MODALITIES>` | Infrared and Visible Image Fusion |
| `<MULTI-FOCUS>` | Multi-focus Image Fusion |
| `<MULTI-EXPOSURE>` | Multi-exposure Image Fusion |

### 3.2 Control Subtasks

| Subtag | Effect |
|--------|--------|
| `<LIGHT++>` | Strong brightness increase |
| `<LIGHT+>` | Slight brightness increase |
| `<LIGHT->` | Slight brightness decrease |
| `<LIGHT-->` | Strong brightness decrease |
| `<CONTRAST+>` | Increase contrast |
| `<CONTRAST->` | Decrease contrast |

> For **M3 training data**, only `[FUSION]` is used without any `<SUBTASK>`.

---

## 4. Image Placeholders

Two special tokens are used to represent the visual inputs:

```text
<img><|image_1|></img>
<img><|image_2|></img>
```

Each image is encoded by a VAE and injected into the DiT latent space alongside the textual tokens.

---

## 5. Standard Prompt Examples


### 5.1 Infrared & Visible Image Fusion (IVIF)

```text
[FUSION]. <MULTI-MODALITIES>. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate a fused image that preserves both the thermal information 
from the infrared modality and the texture details from the visible modality.
```


---

### 5.2 Multi-Focus Image Fusion (MFF)

```text
[FUSION]. <MULTI-FOCUS>. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate the image based on the following requirements: 
Detect sharp regions via Laplacian focus measures, create an all-in-focus mask, 
and seamlessly blend them to produce a tack-sharp photo from foreground to background 
without visible seams.
```


---

### 5.3 Multi-Exposure Image Fusion (MEF)

```text
[FUSION]. <MULTI-EXPOSURE>. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate the image based on the following requirements: 
Merge the well-exposed mid-tones of image 1 with the shadow details of image 2, 
apply locally adaptive tone mapping to preserve texture contrast, 
and ensure natural global luminance.
```


---

### 5.4 M3 Generic Fusion (Self-Supervised)

```text
[FUSION]. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate the image based on the following requirements: 
Extract and fuse high-quality features from both images.
```

---

### 5.5 Text-Controlled Image Fusion (Brightness Control)

```text
[CONTROL]. <LIGHT+>. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate the image based on the following requirements: 
Extract and fuse high-quality features from both images. 
Slightly brighten the critical elements to make them stand out subtly.
```

### 5.6 Segmentation

```text
[SEG]. This is the first image <img><|image_1|></img>, 
and this is the second image <img><|image_2|></img>. 
Please generate the image based on the following requirements: 
Segment the tree in the image.
```

---

## 6. Prompt Robustness

Thanks to large-scale multi-task instruction tuning and M3-based self-supervision, **DiTFuse exhibits strong robustness to linguistic variation**. In practice, the model does not rely on rigid template matching:

> **Arbitrary natural-language variants of the prompt, as long as they express equivalent intent, can still reliably drive correct fusion, control, and segmentation behaviors.**

This ensures strong generalization and real-world usability.
