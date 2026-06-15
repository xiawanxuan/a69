
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


class DefectVisualizer:
    """
    缺陷掩码可视化工具

    支持：
    - 掩码叠加图
    - 缺陷轮廓标注
    - 缺陷坐标点标注
    - 统计信息绘制
    - 批量可视化
    """

    def __init__(self, mask_color=(255, 0, 0), alpha=0.4,
                 contour_color=(0, 255, 255), point_color=(0, 0, 255)):
        """
        Args:
            mask_color: 掩码颜色 (R, G, B)
            alpha: 掩码透明度
            contour_color: 轮廓颜色
            point_color: 点标注颜色
        """
        self.mask_color = mask_color
        self.alpha = alpha
        self.contour_color = contour_color
        self.point_color = point_color

    def create_overlay(self, image, mask, alpha=None):
        """
        创建掩码叠加图

        Args:
            image: 原始图像 (PIL Image 或 numpy array)
            mask: 二值掩码 (numpy array [H, W])
            alpha: 透明度，默认使用初始化值

        Returns:
            PIL.Image: 叠加图
        """
        if alpha is None:
            alpha = self.alpha

        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = image.copy()

        mask = mask.astype(np.uint8)

        # 创建彩色掩码
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = self.mask_color

        # 叠加
        overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

        return Image.fromarray(overlay)

    def draw_contours(self, image, mask, thickness=2):
        """
        在图像上绘制缺陷轮廓

        Args:
            image: 原始图像
            mask: 二值掩码
            thickness: 轮廓线宽度

        Returns:
            PIL.Image: 带轮廓标注的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = image.copy()

        mask = mask.astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 绘制轮廓
        cv2.drawContours(image, contours, -1, self.contour_color, thickness)

        return Image.fromarray(image)

    def draw_defect_points(self, image, defect_points, radius=3, thickness=-1):
        """
        在图像上绘制缺陷坐标点

        Args:
            image: 原始图像
            defect_points: 缺陷点列表 [{"x": x, "y": y, ...}, ...]
            radius: 点半径
            thickness: 线宽，-1 表示填充

        Returns:
            PIL.Image: 带点标注的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = image.copy()

        for point in defect_points:
            x, y = int(point["x"]), int(point["y"])
            cv2.circle(image, (x, y), radius, self.point_color, thickness)

        return Image.fromarray(image)

    def draw_info(self, image, result, position="top-left"):
        """
        在图像上绘制检测信息

        Args:
            image: 原始图像
            result: 推理结果字典
            position: 信息位置 "top-left", "top-right", "bottom-left", "bottom-right"

        Returns:
            PIL.Image: 带信息的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = image.copy()

        # 信息文本
        lines = []
        lines.append(f"Defect: {'Yes' if result['has_defect'] else 'No'}")
        lines.append(f"Score: {result['class_score']:.3f}")
        lines.append(f"Area: {result['defect_area']} px")
        lines.append(f"Ratio: {result['defect_ratio']:.2%}")

        # 绘制背景框
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 20
        padding = 10

        max_text_width = max(
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][0]
            for line in lines
        )
        box_width = max_text_width + 2 * padding
        box_height = len(lines) * line_height + 2 * padding

        h, w = image.shape[:2]

        # 位置
        if "top" in position:
            y = padding
        else:
            y = h - box_height - padding

        if "left" in position:
            x = padding
        else:
            x = w - box_width - padding

        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(
            overlay, (x, y), (x + box_width, y + box_height),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # 绘制文字
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_height - 5
            cv2.putText(
                image, line, (x + padding, text_y),
                font, font_scale, (255, 255, 255), font_thickness
            )

        return Image.fromarray(image)

    def visualize(self, image, result, show_mask=True, show_contours=True,
                  show_points=True, show_info=True):
        """
        完整可视化（叠加所有标注）

        Args:
            image: 原始图像
            result: 推理结果字典
            show_mask: 是否显示掩码叠加
            show_contours: 是否显示轮廓
            show_points: 是否显示坐标点
            show_info: 是否显示信息

        Returns:
            PIL.Image: 可视化结果
        """
        vis_image = image

        if show_mask:
            vis_image = self.create_overlay(vis_image, result["mask"])

        if show_contours:
            vis_image = self.draw_contours(vis_image, result["mask"])

        if show_points and result.get("defect_points"):
            vis_image = self.draw_defect_points(vis_image, result["defect_points"])

        if show_info:
            vis_image = self.draw_info(vis_image, result)

        return vis_image

    def visualize_batch(self, results, output_dir, save_individual=True,
                        save_grid=False, grid_cols=4):
        """
        批量可视化推理结果

        Args:
            results: 推理结果列表
            output_dir: 输出目录
            save_individual: 是否保存单张图片
            save_grid: 是否保存网格图
            grid_cols: 网格列数
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vis_images = []

        for result in results:
            image_path = result["image_path"]
            orig_image = Image.open(image_path).convert("RGB")

            vis_image = self.visualize(orig_image, result)
            vis_images.append(vis_image)

            if save_individual:
                base_name = Path(image_path).stem
                save_path = output_dir / f"{base_name}_vis.png"
                vis_image.save(save_path)

        if save_grid and vis_images:
            grid_image = self._make_grid(vis_images, cols=grid_cols)
            grid_path = output_dir / "grid_visualization.png"
            grid_image.save(grid_path)

    def _make_grid(self, images, cols=4):
        """
        创建网格拼接图
        """
        if not images:
            return None

        # 统一尺寸
        thumb_size = (256, 256)
        thumb_images = [img.resize(thumb_size) for img in images]

        rows = (len(thumb_images) + cols - 1) // cols
        width = cols * thumb_size[0]
        height = rows * thumb_size[1]

        grid = Image.new("RGB", (width, height), color="white")

        for i, img in enumerate(thumb_images):
            row = i // cols
            col = i % cols
            x = col * thumb_size[0]
            y = row * thumb_size[1]
            grid.paste(img, (x, y))

        return grid

    def create_comparison(self, image, pred_mask, gt_mask=None):
        """
        创建预测与真实掩码对比图

        Args:
            image: 原始图像
            pred_mask: 预测掩码
            gt_mask: 真实掩码（可选）

        Returns:
            PIL.Image: 对比图
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        images = [Image.fromarray(image)]

        # 预测掩码叠加
        pred_overlay = self.create_overlay(image, pred_mask)
        pred_overlay = self.draw_contours(pred_overlay, pred_mask)
        images.append(pred_overlay)

        # 真实掩码叠加
        if gt_mask is not None:
            gt_overlay = self.create_overlay(image, gt_mask)
            gt_overlay = self.draw_contours(gt_overlay, gt_mask)
            images.append(gt_overlay)

        # 横向拼接
        widths = [img.width for img in images]
        height = max(img.height for img in images)
        total_width = sum(widths)

        combined = Image.new("RGB", (total_width, height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        return combined
