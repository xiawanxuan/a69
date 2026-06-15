
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from PIL import Image
import cv2

from utils import load_config, get_device
from models import build_model
from data import build_inference_dataset


class PCBSegmentor:
    """
    PCB 缺陷分割推理器

    支持单图和批量推理，输出：
    - 缺陷二值掩码
    - 缺陷像素总面积
    - 短路/缺陷坐标点位
    """

    def __init__(self, checkpoint_path, config=None, device=None):
        """
        Args:
            checkpoint_path: 模型检查点路径
            config: 配置字典或配置文件路径
            device: 计算设备（自动检测）
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 加载配置
        if config is None:
            self.config = checkpoint.get("config", {})
        elif isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config

        # 设备
        if device is None:
            self.device = get_device(self.config)
        else:
            self.device = device

        # 构建模型
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 推理配置
        infer_cfg = self.config.get("inference", {})
        self.threshold = infer_cfg.get("threshold", 0.5)
        self.min_defect_area = infer_cfg.get("min_defect_area", 50)
        self.image_size = tuple(self.config.get("data", {}).get("image_size", [256, 256]))

    @torch.no_grad()
    def predict(self, image_path):
        """
        单图推理

        Args:
            image_path: 图片路径

        Returns:
            dict: 推理结果，包含：
                - mask: 二值掩码 numpy array [H, W]
                - mask_prob: 概率掩码 numpy array [H, W]
                - defect_area: 缺陷像素面积
                - defect_ratio: 缺陷占比
                - defect_points: 缺陷坐标点列表 [(x, y), ...]
                - has_defect: 是否有缺陷
                - class_score: 图像级分类分数
        """
        # 加载原始图像
        orig_img = Image.open(image_path).convert("RGB")
        orig_size = orig_img.size  # (width, height)

        # 构建数据加载器（单张）
        dataloader = build_inference_dataset(
            [image_path], self.config, self.image_size
        )

        # 推理
        for batch in dataloader:
            images = batch["image"].to(self.device)
            outputs = self.model(images)

            # 获取概率掩码
            mask_prob = torch.sigmoid(outputs["mask"])
            mask_prob = mask_prob[0, 0].cpu().numpy()

            # 分类分数
            class_score = torch.sigmoid(outputs["class_logit"])[0, 0].item()

        # 调整回原始尺寸
        mask_prob = cv2.resize(mask_prob, orig_size, interpolation=cv2.INTER_LINEAR)

        # 二值化
        binary_mask = (mask_prob > self.threshold).astype(np.uint8) * 255

        # 连通区域分析，过滤小面积缺陷
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # 过滤小区域
        filtered_mask = np.zeros_like(binary_mask)
        defect_points = []
        total_defect_area = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_defect_area:
                filtered_mask[labels == i] = 255
                total_defect_area += area

                # 收集缺陷区域内的代表性点（中心点 + 轮廓点）
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                defect_points.append({"x": cx, "y": cy, "area": int(area)})

                # 轮廓点
                contours, _ = cv2.findContours(
                    (labels == i).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    # 每隔几个点取一个轮廓点
                    contour = contours[0].reshape(-1, 2)
                    step = max(1, len(contour) // 10)
                    for pt in contour[::step]:
                        defect_points.append({
                            "x": int(pt[0]),
                            "y": int(pt[1]),
                            "type": "contour"
                        })

        # 计算缺陷占比
        h, w = mask_prob.shape
        defect_ratio = total_defect_area / (h * w) if h * w > 0 else 0

        has_defect = total_defect_area > 0 and class_score > self.threshold

        return {
            "mask": filtered_mask,
            "mask_prob": mask_prob,
            "defect_area": int(total_defect_area),
            "defect_ratio": float(defect_ratio),
            "defect_points": defect_points,
            "has_defect": bool(has_defect),
            "class_score": float(class_score),
            "image_path": str(image_path),
            "image_size": {"width": w, "height": h},
        }

    @torch.no_grad()
    def predict_batch(self, image_paths):
        """
        批量推理

        Args:
            image_paths: 图片路径列表

        Returns:
            list: 每张图片的推理结果列表
        """
        dataloader = build_inference_dataset(
            image_paths, self.config, self.image_size
        )

        results = []
        idx = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            batch_size = images.size(0)

            outputs = self.model(images)
            mask_probs = torch.sigmoid(outputs["mask"])
            class_scores = torch.sigmoid(outputs["class_logit"])

            for i in range(batch_size):
                if idx >= len(image_paths):
                    break

                # 加载原始图像获取尺寸
                orig_img = Image.open(image_paths[idx]).convert("RGB")
                orig_size = orig_img.size  # (width, height)

                mask_prob = mask_probs[i, 0].cpu().numpy()
                mask_prob = cv2.resize(mask_prob, orig_size, interpolation=cv2.INTER_LINEAR)

                # 二值化
                binary_mask = (mask_prob > self.threshold).astype(np.uint8) * 255

                # 连通区域分析
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    binary_mask, connectivity=8
                )

                filtered_mask = np.zeros_like(binary_mask)
                defect_points = []
                total_defect_area = 0

                for j in range(1, num_labels):
                    area = stats[j, cv2.CC_STAT_AREA]
                    if area >= self.min_defect_area:
                        filtered_mask[labels == j] = 255
                        total_defect_area += area

                        cx, cy = int(centroids[j][0]), int(centroids[j][1])
                        defect_points.append({"x": cx, "y": cy, "area": int(area)})

                h, w = mask_prob.shape
                defect_ratio = total_defect_area / (h * w) if h * w > 0 else 0
                class_score = class_scores[i, 0].item()
                has_defect = total_defect_area > 0 and class_score > self.threshold

                results.append({
                    "mask": filtered_mask,
                    "mask_prob": mask_prob,
                    "defect_area": int(total_defect_area),
                    "defect_ratio": float(defect_ratio),
                    "defect_points": defect_points,
                    "has_defect": bool(has_defect),
                    "class_score": float(class_score),
                    "image_path": str(image_paths[idx]),
                    "image_size": {"width": w, "height": h},
                })

                idx += 1

        return results

    def predict_folder(self, folder_path, output_dir=None, save_mask=True,
                       save_overlay=True):
        """
        对文件夹中的所有图片进行批量推理

        Args:
            folder_path: 图片文件夹路径
            output_dir: 输出目录
            save_mask: 是否保存掩码
            save_overlay: 是否保存叠加图

        Returns:
            dict: 统计结果
        """
        folder_path = Path(folder_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        image_paths = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_paths:
            raise FileNotFoundError(f"在 {folder_path} 中没有找到图片")

        print(f"找到 {len(image_paths)} 张图片，开始推理...")

        results = self.predict_batch(image_paths)

        # 统计
        total = len(results)
        defect_count = sum(1 for r in results if r["has_defect"])
        avg_defect_area = np.mean([r["defect_area"] for r in results]) if results else 0

        stats = {
            "total_images": total,
            "defect_images": defect_count,
            "normal_images": total - defect_count,
            "defect_ratio": defect_count / total if total > 0 else 0,
            "avg_defect_area": float(avg_defect_area),
        }

        # 保存结果
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存掩码和叠加图
            for result in results:
                img_path = Path(result["image_path"])
                base_name = img_path.stem

                if save_mask:
                    mask_path = output_dir / f"{base_name}_mask.png"
                    Image.fromarray(result["mask"]).save(mask_path)

                if save_overlay:
                    overlay = self._create_overlay(
                        Image.open(result["image_path"]).convert("RGB"),
                        result["mask"]
                    )
                    overlay_path = output_dir / f"{base_name}_overlay.png"
                    overlay.save(overlay_path)

            # 保存统计结果
            import json
            stats_path = output_dir / "statistics.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            print(f"结果已保存到: {output_dir}")

        print(f"\n推理完成：")
        print(f"  总图片数: {total}")
        print(f"  缺陷图片: {defect_count}")
        print(f"  正常图片: {total - defect_count}")
        print(f"  缺陷率: {stats['defect_ratio']:.2%}")
        print(f"  平均缺陷面积: {avg_defect_area:.1f} 像素")

        return {
            "results": results,
            "statistics": stats,
        }

    def _create_overlay(self, original_img, mask, color=(255, 0, 0), alpha=0.5):
        """
        创建掩码叠加图

        Args:
            original_img: 原始 PIL Image
            mask: 二值掩码 numpy array
            color: 叠加颜色 (R, G, B)
            alpha: 透明度

        Returns:
            PIL.Image: 叠加图
        """
        orig = np.array(original_img)
        mask_colored = np.zeros_like(orig)
        mask_colored[mask > 0] = color

        overlay = cv2.addWeighted(orig, 1 - alpha, mask_colored, alpha, 0)
        return Image.fromarray(overlay)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PCB 批量推理脚本")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--input", type=str, required=True,
                        help="输入图片路径或文件夹")
    parser.add_argument("--output", type=str, default="outputs/inference",
                        help="输出目录")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--no_save_mask", action="store_true",
                        help="不保存掩码图")
    parser.add_argument("--no_save_overlay", action="store_true",
                        help="不保存叠加图")
    parser.add_argument("--threshold", type=float, default=None,
                        help="分割阈值")
    parser.add_argument("--min_area", type=int, default=None,
                        help="最小缺陷面积")

    args = parser.parse_args()

    # 创建推理器
    segmentor = PCBSegmentor(args.checkpoint, args.config)

    # 覆盖参数
    if args.threshold is not None:
        segmentor.threshold = args.threshold
    if args.min_area is not None:
        segmentor.min_defect_area = args.min_area

    # 推理
    input_path = Path(args.input)
    if input_path.is_dir():
        segmentor.predict_folder(
            input_path,
            output_dir=args.output,
            save_mask=not args.no_save_mask,
            save_overlay=not args.no_save_overlay,
        )
    else:
        result = segmentor.predict(str(input_path))
        print(f"\n推理结果:")
        print(f"  有缺陷: {result['has_defect']}")
        print(f"  分类分数: {result['class_score']:.4f}")
        print(f"  缺陷面积: {result['defect_area']} 像素")
        print(f"  缺陷占比: {result['defect_ratio']:.2%}")
        print(f"  缺陷点数: {len(result['defect_points'])}")

        # 保存结果
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_save_mask:
            mask_path = output_dir / f"{input_path.stem}_mask.png"
            Image.fromarray(result["mask"]).save(mask_path)
            print(f"  掩码已保存: {mask_path}")

        if not args.no_save_overlay:
            overlay = segmentor._create_overlay(
                Image.open(input_path).convert("RGB"),
                result["mask"]
            )
            overlay_path = output_dir / f"{input_path.stem}_overlay.png"
            overlay.save(overlay_path)
            print(f"  叠加图已保存: {overlay_path}")


if __name__ == "__main__":
    main()
