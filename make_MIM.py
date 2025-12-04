#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import math
import json
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------
# 随机强度的降质操作
# ---------------------------

def apply_mask_noise(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """将区域用均匀噪声覆盖（噪声mask）"""
    h_clip = min(h, img.shape[0] - y)
    w_clip = min(w, img.shape[1] - x)
    noise = np.random.randint(0, 256, size=(h_clip, w_clip, img.shape[2]), dtype=np.uint8)
    img[y:y+h_clip, x:x+w_clip] = noise

def apply_blur(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """对区域进行模糊（随机选择高斯/均值/中值 + 随机核）"""
    h_clip = min(h, img.shape[0] - y)
    w_clip = min(w, img.shape[1] - x)
    roi = img[y:y+h_clip, x:x+w_clip]

    # 随机核大小（奇数）和方式
    k_choices = [3, 5, 7, 9, 11]
    k = random.choice(k_choices)
    mode = random.choice(["gaussian", "box", "median"])

    if mode == "gaussian":
        sigma = random.uniform(0.8, 3.0)
        blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    elif mode == "box":
        blurred = cv2.blur(roi, (k, k), borderType=cv2.BORDER_REFLECT_101)
    else:  # median
        # 中值滤波只支持单通道/三通道 uint8，满足
        blurred = cv2.medianBlur(roi, k)

    img[y:y+h_clip, x:x+w_clip] = blurred

def apply_add_noise(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """对区域加噪（随机高斯或椒盐），强度适中"""
    h_clip = min(h, img.shape[0] - y)
    w_clip = min(w, img.shape[1] - x)
    roi = img[y:y+h_clip, x:x+w_clip]

    mode = random.choice(["gaussian", "sp"])  # 高斯 or 椒盐
    if mode == "gaussian":
        # 标准差 10-35（0~255量纲）
        sigma = random.uniform(10.0, 35.0)
        noise = np.random.normal(0, sigma, roi.shape).astype(np.float32)
        noisy = roi.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        img[y:y+h_clip, x:x+w_clip] = noisy
    else:
        # 椒盐噪声比例 0.01 - 0.03
        prob = random.uniform(0.01, 0.03)
        noisy = roi.copy()
        # 椒点
        num_salt = int(prob * roi.shape[0] * roi.shape[1] / 2)
        coords = (np.random.randint(0, roi.shape[0], num_salt),
                  np.random.randint(0, roi.shape[1], num_salt))
        noisy[coords[0], coords[1]] = 255
        # 盐点
        num_pepper = int(prob * roi.shape[0] * roi.shape[1] / 2)
        coords = (np.random.randint(0, roi.shape[0], num_pepper),
                  np.random.randint(0, roi.shape[1], num_pepper))
        noisy[coords[0], coords[1]] = 0
        img[y:y+h_clip, x:x+w_clip] = noisy

def degrade_once(img: np.ndarray, rect: Tuple[int,int,int,int], op: str) -> None:
    x, y, w, h = rect
    if op == "mask":
        apply_mask_noise(img, x, y, w, h)
    elif op == "blur":
        apply_blur(img, x, y, w, h)
    elif op == "noise":
        apply_add_noise(img, x, y, w, h)
    else:
        raise ValueError(f"未知降质类型: {op}")

# ---------------------------
# 分块与流程
# ---------------------------

def get_blocks(W: int, H: int, block: int) -> List[Tuple[int,int,int,int]]:
    """按照给定 block（正方形边长），生成覆盖全图的块列表。
    边界不足整块时，最后一列/行用裁剪后的矩形（可能非正方形）。"""
    rects = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            rects.append((x, y, block, block))
            x += block
        y += block
    return rects

def choose_ops_for_blocks(n_blocks: int, degrades: List[str]) -> List[str]:
    """为每个块选择一个操作；若只给一个选项则不随机"""
    if len(degrades) == 1:
        return [degrades[0]] * n_blocks
    return [random.choice(degrades) for _ in range(n_blocks)]

def process_image(img_path: Path, out1: Path, out2: Path,
                  block_sizes=(32, 64, 128),
                  degrades=("mask", "noise", "blur"),
                  both_degrade=False) -> dict:
    """返回元信息（方便调试/复现），并在 out1/out2 写出两张图"""
    meta = {
        "filename": img_path.name,
        "block_size": None,
        "num_blocks": 0,
        "both_degrade": both_degrade,
        "both_ids": [],
        "ops": [],  # per block op
    }

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取失败: {img_path}")

    H, W = img.shape[:2]
    block = random.choice(block_sizes)
    meta["block_size"] = block

    rects = get_blocks(W, H, block)
    meta["num_blocks"] = len(rects)

    # 两张复制图
    img1 = img.copy()
    img2 = img.copy()

    # 为每个块选择操作
    ops = choose_ops_for_blocks(len(rects), list(degrades))
    meta["ops"] = ops

    # 选取 1/4 的块 id 用于 both degrade
    both_ids = set()
    if both_degrade and len(rects) > 0:
        k = max(1, len(rects) // 4)  # 至少选择 1 个
        both_ids = set(random.sample(range(len(rects)), k))
        meta["both_ids"] = sorted(list(both_ids))

    # 逐块应用
    for idx, rect in enumerate(rects):
        op = ops[idx]
        if idx in both_ids:
            # 两张图都处理（同一种操作）
            degrade_once(img1, rect, op)
            degrade_once(img2, rect, op)
        else:
            # 两张中随机一张处理
            if random.random() < 0.5:
                degrade_once(img1, rect, op)
            else:
                degrade_once(img2, rect, op)

    # 保存
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    ok1 = cv2.imwrite(str(out1), img1)
    ok2 = cv2.imwrite(str(out2), img2)
    if not (ok1 and ok2):
        raise RuntimeError(f"写出失败: {out1} 或 {out2}")
    return meta

# ---------------------------
# 主程序
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="按块随机降质，生成成对图像（img1/img2）")
    parser.add_argument("--input", type=str, required=True, help="输入图像文件夹")
    parser.add_argument("--output", type=str, required=True, help="输出根目录（将含 img1/ 与 img2/）")
    parser.add_argument("--degrades", type=str, nargs="*", default=None,
                        help="降质类型集合：mask noise blur（不传则三种都启用）")
    parser.add_argument("--both-degrade", action="store_true",
                        help="启用后，随机 1/4 块 id 在两张图上都应用降质")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    parser.add_argument("--block-sizes", type=int, nargs="*", default=[32, 64, 128],
                        help="备选的正方形块边长，默认 32 64 128")
    parser.add_argument("--save-meta", action="store_true",
                        help="将每张图的元信息记录到 metadata.jsonl")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    in_dir = Path(args.input)
    out_root = Path(args.output)
    out1_dir = out_root / "img1"
    out2_dir = out_root / "img2"

    if not in_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{in_dir}")

    # 降质集合
    if args.degrades is None or len(args.degrades) == 0:
        degrades = ["mask", "noise", "blur"]
    else:
        degrades = [d.lower() for d in args.degrades]
        for d in degrades:
            if d not in {"mask", "noise", "blur"}:
                raise ValueError(f"不支持的降质类型: {d}")

    metas = []
    files = [p for p in sorted(in_dir.iterdir())
             if p.is_file() and p.suffix.lower() in IMG_EXT]

    for p in tqdm(files, desc="Processing"):
        out1 = out1_dir / p.name
        out2 = out2_dir / p.name
        try:
            meta = process_image(
                p, out1, out2,
                block_sizes=tuple(args.block_sizes),
                degrades=tuple(degrades),
                both_degrade=args.both_degrade
            )
            metas.append(meta)
        except Exception as e:
            print(f"[跳过] {p.name}: {e}")

    if args.save_meta:
        meta_path = out_root / "metadata.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"已写出元信息: {meta_path}")

    print(f"完成。输出目录：\n  {out1_dir}\n  {out2_dir}")

if __name__ == "__main__":
    main()



# python make_MIM.py \
#   --input /mnt/shared-storage-user/lijiayang1/DiTFuse/data/train2017 \
#   --output /mnt/shared-storage-user/lijiayang1/DiTFuse/data/train2017_ONLY_noise \
#   --degrades mask noise blur \
#   --both-degrade \
#   --seed 42

#   python make_MIM.py \
#   --input /mnt/shared-storage-user/lijiayang1/DiTFuse/data/train2017 \
#   --output /mnt/shared-storage-user/lijiayang1/DiTFuse/data/train2017_ONLY_noise \
#   --degrades noise \
#   --seed 42
