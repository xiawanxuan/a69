
import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


def split_dataset(
    image_dir: str,
    label_dir: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = None,
) -> Dict[str, List[str]]:
    """
    划分数据集为训练集、验证集、测试集

    支持两种模式：
    1. 有像素级标签：label_dir 存在掩码图片
    2. 只有图像级标签：图片文件名包含 "defect" 或 "normal" 等关键词

    Args:
        image_dir: 图像目录路径
        label_dir: 标签目录路径（可选，无像素标签时为 None）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        output_dir: 输出目录，保存划分结果的 JSON 文件

    Returns:
        dict: 包含 train/val/test 三个列表的字典，每个元素是 (image_path, label_path, image_label)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "数据集比例之和必须为 1.0"

    random.seed(seed)
    np.random.seed(seed)

    image_dir = Path(image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # 收集所有图片
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise FileNotFoundError(f"在 {image_dir} 中没有找到图片文件")

    # 判断是否有像素级标签
    has_pixel_labels = label_dir is not None and Path(label_dir).exists()

    # 生成图像级标签
    samples = []
    for img_path in image_files:
        # 根据文件名判断是否有缺陷
        stem = img_path.stem.lower()
        has_defect = 1 if (
            "defect" in stem or "short" in stem or "crack" in stem
            or "bad" in stem or "fail" in stem or "ng" in stem
        ) else 0

        # 如果有标签目录，查找对应的标签文件
        label_path = None
        if has_pixel_labels:
            for ext in image_extensions:
                candidate = Path(label_dir) / f"{img_path.stem}{ext}"
                if candidate.exists():
                    label_path = str(candidate)
                    break

        samples.append({
            "image": str(img_path),
            "label": label_path,
            "image_label": has_defect,
        })

    # 按类别分层抽样
    positive_samples = [s for s in samples if s["image_label"] == 1]
    negative_samples = [s for s in samples if s["image_label"] == 0]

    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    def split_list(lst, tr, vr):
        n = len(lst)
        n_train = int(n * tr)
        n_val = int(n * vr)
        return (
            lst[:n_train],
            lst[n_train:n_train + n_val],
            lst[n_train + n_val:]
        )

    pos_train, pos_val, pos_test = split_list(positive_samples, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_list(negative_samples, train_ratio, val_ratio)

    train_set = pos_train + neg_train
    val_set = pos_val + neg_val
    test_set = pos_test + neg_test

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    result = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    # 打印统计信息
    print(f"数据集划分完成：")
    print(f"  训练集: {len(train_set)} 张 (正样本 {len(pos_train)}, 负样本 {len(neg_train)})")
    print(f"  验证集: {len(val_set)} 张 (正样本 {len(pos_val)}, 负样本 {len(neg_val)})")
    print(f"  测试集: {len(test_set)} 张 (正样本 {len(pos_test)}, 负样本 {len(neg_test)})")
    print(f"  像素级标签: {'有' if has_pixel_labels else '无（弱监督模式）'}")

    # 保存划分结果
    if output_dir is not None:
        output_path = Path(output_dir) / "dataset_split.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"划分结果已保存到: {output_path}")

    return result


def load_dataset_split(split_file: str) -> Dict[str, List[Dict]]:
    """
    加载已保存的数据集划分

    Args:
        split_file: 划分结果 JSON 文件路径

    Returns:
        dict: 数据集划分字典
    """
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCB 数据集划分脚本")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录")
    parser.add_argument("--label_dir", type=str, default=None, help="标签目录（可选）")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="datasets", help="输出目录")

    args = parser.parse_args()

    split_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )
