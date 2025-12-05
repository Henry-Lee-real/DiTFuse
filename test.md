## 📁 Test File Structure

For large-scale or batch testing, we recommend organizing the dataset in the following structure:

```

dataset_root/
│── image1/   # Visible / Far / Over  (V / F / O)
│── image2/   # Infrared / Near / Under (IR / N / U)

```

You may use abbreviations such as:

- **image1:** V / F / O  
- **image2:** IR / N / U  

> A visual illustration of this structure is shown below:

```

dataset_root
├── image1
│     ├── 0001.png
│     ├── 0002.png
│     └── ...
└── image2
├── 0001.png
├── 0002.png
└── ...

```

Each pair must share the **same filename** across the two folders.




## 🚀 Run Guide

This project provides three testing modes: **single image pair**, **batch processing**, and **multi-prompt testing**.

### Single Pair Testing

```bash
python test.py \
  --checkpoint_path /path/to/ckpt/omnigen \
  --lora_path /path/to/ckpt/lora/ \
  --output_dir /path/to/out/ \
  --mode single \
  --image1 path/to/img1.png \
  --image2 path/to/img2.png \
  --prompt_single "XXX"
```


## Batch Testing (Multiple Pairs)

```bash
python test.py \
  --checkpoint_path /path/to/ckpt/omnigen \
  --lora_path /path/to/ckpt/lora/ \
  --output_dir /path/to/out/ \
  --mode batch \
  --image_dir /path/to/batch_images \
  --prompt_batch "XXX"
```

Folder structure requirements for `image_dir` are described above.



## Multi-Prompt Testing

```bash
python test.py \
  --checkpoint_path /path/to/ckpt/omnigen \
  --lora_path /path/to/ckpt/lora/ \
  --output_dir /path/to/out/ \
  --mode prompt \
  --image1 path/to/img1.png \
  --image2 path/to/img2.png \
  --prompt_file /path/to/prompt.txt
```

Each line in `prompt.txt` will generate a separate fused output.


## ⚙️ Advanced Usage

### **guidance_scale**

Controls the strength of **text instruction** guidance.

* Higher values → model follows text prompts more strictly.
* Lower values → fusion relies more on image content.

### **img_guidance_scale**

Controls the influence of the **input images** during the fusion process.

* Higher values → the fused image preserves more original visual structures.
* Lower values → the model is freer to modify appearance based on the prompt.




If you want，我可以继续为你把这一节做成 **带图标、分栏布局、甚至增强视觉效果的专业 README 风格**。
