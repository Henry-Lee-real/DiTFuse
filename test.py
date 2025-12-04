import os
import io
import argparse
import shutil
from typing import Tuple

from PIL import Image
from OmniGen import OmniGenPipeline


# ---------------------- image utilities ---------------------- #

def prepare_image_for_model(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """
    Prepare an image for the model *in memory* (no disk modification).

    New rules:
      1) Always rescale the image so that its longest side becomes `max_side`
         (i.e., may downscale or upscale, aspect ratio preserved).
      2) Afterwards, both width and height are floored to the nearest
         multiple of 16.
      3) If nothing changes after these operations, return the original image.

    Args:
        img: PIL image.
        max_side: target length for the longest side after resizing.

    Returns:
        A resized PIL image whose longest side is approximately `max_side`
        (and both dimensions are multiples of 16).
    """

    def floor_to_16(x: int) -> int:
        return max(16, (x // 16) * 16)

    w, h = img.size
    longest = max(w, h)

    if longest == 0:
        return img  # 避免除零，极端情况兜底

    # Step 1: 强制缩放到最长边为 max_side（允许上采样 / 下采样）
    scale = float(max_side) / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Step 2: floor each dimension to multiple of 16
    new_w = floor_to_16(new_w)
    new_h = floor_to_16(new_h)

    # If nothing changes, return original
    if new_w == w and new_h == h:
        return img

    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_bytesio(img: Image.Image) -> io.BytesIO:
    """
    Convert a PIL image into an in-memory file object.

    OmniGen's processor calls `Image.open(x)`. As long as `x` has a `.read()`
    method (like BytesIO), this works.

    NOTE: 每次使用前都要重新 new 一个 BytesIO，避免读指针在末尾。
    """
    bio = io.BytesIO()
    img.save(bio, format="PNG")  # PNG is lossless; you can change if needed.
    bio.seek(0)
    return bio


def is_valid_image(image_path: str) -> bool:
    """Check whether the given path points to a valid image file."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ---------------------- pipeline helpers ---------------------- #

def load_pipeline(checkpoint_path: str, lora_path: str) -> OmniGenPipeline:
    """
    Load OmniGen pipeline and merge LoRA weights once.

    checkpoint_path: base OmniGen model.
    lora_path:       LoRA weights to be merged into the base model.
    """
    pipe = OmniGenPipeline.from_pretrained(checkpoint_path)
    pipe.merge_lora(lora_path)
    return pipe


def process_single_image(
    pipe: OmniGenPipeline,
    prompt: str,
    image_path_1: str,
    image_path_2: str,
    output_path: str,
    max_side: int = 256,
    guidance_scale: float = 2.0,
    img_guidance_scale: float = 1.6,
    seed: int = 0,
) -> None:
    """
    Process a single (image_1, image_2) pair and save the fused result.
    All resizing happens in memory; original files remain unchanged.

    Args:
        pipe: OmniGenPipeline already loaded with LoRA.
        prompt: text prompt describing how to fuse / interpret image_1 and image_2.
        image_path_1: path to the first input image.
        image_path_2: path to the second input image.
        output_path: where to save the fused result.
        max_side: target longest side passed to `prepare_image_for_model`.
        guidance_scale: text CFG scale.
        img_guidance_scale: image CFG scale.
        seed: random seed for generation.
    """
    # Load original images
    img1_raw = Image.open(image_path_1).convert("RGB")
    img2_raw = Image.open(image_path_2).convert("RGB")

    # Prepare for model (rescale + /16)
    img1 = prepare_image_for_model(img1_raw, max_side=max_side)
    img2 = prepare_image_for_model(img2_raw, max_side=max_side)

    width, height = img1.size

    # Convert to in-memory file objects so that OmniGen can `Image.open()` them
    img1_file = image_to_bytesio(img1)
    img2_file = image_to_bytesio(img2)

    images = pipe(
        prompt=prompt,
        input_images=[img1_file, img2_file],
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        img_guidance_scale=img_guidance_scale,
        separate_cfg_infer=True,
        seed=seed,
    )

    images[0].save(output_path)
    print(f"[single] Saved: {output_path}")


def process_batch_images(
    pipe: OmniGenPipeline,
    prompt: str,
    image_dir: str,
    output_dir: str,
    max_side: int = 256,
    guidance_scale: float = 2.0,
    img_guidance_scale: float = 1.6,
    seed: int = 3047,
) -> None:
    """
    Batch mode for two-input fusion.

    Directory layout assumption:
        image_dir/
            image1/   # all first-modality images
            image2/   # all second-modality images
        - image1/ and image2/ should contain files with identical names,
          e.g., image1/0001.png and image2/0001.png form a pair.

    Only .png / .jpg files in `image1/` are iterated and processed.

    Args:
        pipe: OmniGenPipeline already loaded with LoRA.
        prompt: text prompt applied to every (image1, image2) pair.
        image_dir: root directory containing `image1` and `image2` subfolders.
        output_dir: where to save batch outputs.
        max_side: target longest side passed to `prepare_image_for_model`.
        guidance_scale: text CFG scale.
        img_guidance_scale: image CFG scale.
        seed: random seed for generation.
    """
    vi_dir = os.path.join(image_dir, "image1")
    ir_dir = os.path.join(image_dir, "image2")

    vi_images = [
        f for f in os.listdir(vi_dir)
        if f.lower().endswith((".png", ".jpg"))
    ]

    for vi_image in vi_images:
        image_path_1 = os.path.join(vi_dir, vi_image)
        image_path_2 = os.path.join(ir_dir, vi_image)

        if not os.path.exists(image_path_2):
            print(f"[batch] Skipping {vi_image}: no matching second image.")
            continue

        if not is_valid_image(image_path_1) or not is_valid_image(image_path_2):
            print(f"[batch] Skipping invalid images: {image_path_1}, {image_path_2}")
            continue

        # Load & prepare images only in memory
        img1_raw = Image.open(image_path_1).convert("RGB")
        img2_raw = Image.open(image_path_2).convert("RGB")

        img1 = prepare_image_for_model(img1_raw, max_side=max_side)
        img2 = prepare_image_for_model(img2_raw, max_side=max_side)

        width, height = img1.size

        # Each call must get a fresh BytesIO (read pointer at 0)
        img1_file = image_to_bytesio(img1)
        img2_file = image_to_bytesio(img2)

        image_name_1 = os.path.splitext(os.path.basename(image_path_1))[0]

        images = pipe(
            prompt=prompt,
            input_images=[img1_file, img2_file],
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            separate_cfg_infer=True,
            seed=seed,
        )

        out_path = os.path.join(output_dir, f"{image_name_1}.png")
        images[0].save(out_path)
        print(f"[batch] Saved: {out_path}")


def process_prompts(
    pipe: OmniGenPipeline,
    prompt_file: str,
    image_path_1: str,
    image_path_2: str,
    output_dir: str,
    max_side: int = 256,
    guidance_scale: float = 2.5,
    img_guidance_scale: float = 2.0,
    seed: int = 42,
) -> None:
    """
    Prompt mode:
      - Read prompts line-by-line from `prompt_file`;
      - Apply all prompts to the same (image_1, image_2) pair;
      - Save results as <base_name>_<line_idx>.png;
      - Copy prompt_file into output_dir for record (for later inspection).

    Args:
        pipe: OmniGenPipeline already loaded with LoRA.
        prompt_file: text file, one prompt per non-empty line.
        image_path_1: path to the first input image.
        image_path_2: path to the second input image.
        output_dir: where to save all generated images.
        max_side: target longest side passed to `prepare_image_for_model`.
        guidance_scale: text CFG scale.
        img_guidance_scale: image CFG scale.
        seed: random seed for generation.
    """
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(prompt_file, os.path.join(output_dir, os.path.basename(prompt_file)))

    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Load & prepare once; reuse geometrically for all prompts
    img1_raw = Image.open(image_path_1).convert("RGB")
    img2_raw = Image.open(image_path_2).convert("RGB")

    img1 = prepare_image_for_model(img1_raw, max_side=max_side)
    img2 = prepare_image_for_model(img2_raw, max_side=max_side)

    width, height = img1.size
    base_name = os.path.splitext(os.path.basename(image_path_1))[0]

    for idx, prompt in enumerate(lines, start=1):
        # IMPORTANT: new BytesIO per call
        img1_file = image_to_bytesio(img1)
        img2_file = image_to_bytesio(img2)

        images = pipe(
            prompt=prompt,
            input_images=[img1_file, img2_file],
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            separate_cfg_infer=True,
            seed=seed,
        )
        out_name = f"{base_name}_{idx}.png"
        out_path = os.path.join(output_dir, out_name)
        images[0].save(out_path)
        print(f"[prompt] Saved: {out_path} (line {idx})")


# ---------------------- CLI & main ---------------------- #

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command-line usage.
    所有可调参数集中在这里。
    """
    parser = argparse.ArgumentParser(
        description="Process images with OmniGenPipeline (single / batch / prompt)."
    )

    # 基本模式
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "prompt"],
        required=True,
        help="Processing mode.",
    )

    # 路径相关
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/path/to/ckpt/omnigen",
        help="Path to OmniGen checkpoint.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/path/to/ckpt/lora/",
        help="Path to LoRA weights to be merged.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/out/",
        help="Directory to save outputs.",
    )

    # 通用图像尺寸（现在表示“目标最长边”，而不是“上限”）
    parser.add_argument(
        "--max_side",
        type=int,
        default=1024,
        help=(
            "Target length of the longest image side. "
            "All images are rescaled so that their longest side is ~max_side, "
            "then both dimensions are floored to a multiple of 16."
        ),
    )

    # batch 模式用的数据目录
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/path/to/batch_images",
        help=(
            "Root directory for 'batch' mode; expected to contain subfolders "
            "'image1' (first input stream) and 'image2' (second input stream) "
            "with matching filenames."
        ),
    )

    # single / prompt 模式的两张输入图
    parser.add_argument(
        "--image1",
        type=str,
        default="/path/to/image1.png",
        help=(
            "Path to the first input image for 'single' and 'prompt' modes. "
            "The semantic role of this image (e.g., visible / far / over) "
            "should be clarified in the text prompt if needed."
        ),
    )
    parser.add_argument(
        "--image2",
        type=str,
        default="/path/to/image2.png",
        help=(
            "Path to the second input image for 'single' and 'prompt' modes. "
            "Its role (e.g., infrared / near / under) can also be specified "
            "in the prompt for better fusion control."
        ),
    )

    # prompt 模式的 prompt 文件
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/path/to/prompt.txt",
        help="Text file with one prompt per line (for 'prompt' mode).",
    )

    # 文本 prompt（single / batch）
    parser.add_argument(
        "--prompt_single",
        type=str,
        default="XXX",
        help="Text prompt for 'single' mode.",
    )
    parser.add_argument(
        "--prompt_batch",
        type=str,
        default="XXX",
        help="Text prompt for 'batch' mode.",
    )

    # 不同模式的 scale & seed
    parser.add_argument(
        "--guidance_scale_single",
        type=float,
        default=2.0,
        help="Text guidance scale (CFG) for 'single' mode.",
    )
    parser.add_argument(
        "--img_guidance_scale_single",
        type=float,
        default=1.6,
        help="Image guidance scale (CFG) for 'single' mode.",
    )
    parser.add_argument(
        "--seed_single",
        type=int,
        default=0,
        help="Random seed for 'single' mode.",
    )

    parser.add_argument(
        "--guidance_scale_batch",
        type=float,
        default=2.0,
        help="Text guidance scale (CFG) for 'batch' mode.",
    )
    parser.add_argument(
        "--img_guidance_scale_batch",
        type=float,
        default=1.6,
        help="Image guidance scale (CFG) for 'batch' mode.",
    )
    parser.add_argument(
        "--seed_batch",
        type=int,
        default=3047,
        help="Random seed for 'batch' mode.",
    )

    parser.add_argument(
        "--guidance_scale_prompt",
        type=float,
        default=2.5,
        help="Text guidance scale (CFG) for 'prompt' mode.",
    )
    parser.add_argument(
        "--img_guidance_scale_prompt",
        type=float,
        default=2.0,
        help="Image guidance scale (CFG) for 'prompt' mode.",
    )
    parser.add_argument(
        "--seed_prompt",
        type=int,
        default=42,
        help="Random seed for 'prompt' mode.",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    # Paths & output directory from args
    checkpoint_path = args.checkpoint_path
    lora_path = args.lora_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # For naming only
    lora_name = os.path.basename(lora_path.rstrip("/"))

    # Load pipeline once
    pipe = load_pipeline(checkpoint_path, lora_path)

    if args.mode == "single":
        prompt = args.prompt_single

        image_name_1 = os.path.splitext(os.path.basename(args.image1))[0]
        output_path = os.path.join(
            output_dir, f"{lora_name}_{image_name_1}_test_single.png"
        )

        process_single_image(
            pipe,
            prompt=prompt,
            image_path_1=args.image1,
            image_path_2=args.image2,
            output_path=output_path,
            max_side=args.max_side,
            guidance_scale=args.guidance_scale_single,
            img_guidance_scale=args.img_guidance_scale_single,
            seed=args.seed_single,
        )

    elif args.mode == "batch":
        prompt = args.prompt_batch

        process_batch_images(
            pipe,
            prompt=prompt,
            image_dir=args.image_dir,
            output_dir=output_dir,
            max_side=args.max_side,
            guidance_scale=args.guidance_scale_batch,
            img_guidance_scale=args.img_guidance_scale_batch,
            seed=args.seed_batch,
        )

    elif args.mode == "prompt":
        process_prompts(
            pipe,
            prompt_file=args.prompt_file,
            image_path_1=args.image1,
            image_path_2=args.image2,
            output_dir=output_dir,
            max_side=args.max_side,
            guidance_scale=args.guidance_scale_prompt,
            img_guidance_scale=args.img_guidance_scale_prompt,
            seed=args.seed_prompt,
        )


if __name__ == "__main__":
    main()
