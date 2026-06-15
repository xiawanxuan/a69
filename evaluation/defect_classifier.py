
import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class DefectRegion:
    """
    单个缺陷区域的形态学描述
    """
    label_id: int
    area: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    width: int
    height: int
    aspect_ratio: float
    perimeter: float
    rectangularity: float
    solidity: float
    eccentricity: float
    skeleton_length: float
    classification: str = "unknown"
    confidence: float = 0.0
    contour_points: List[Tuple[int, int]] = field(default_factory=list)


class DefectClassifier:
    """
    PCB 缺陷形态学分类器

    基于缺陷区域的形状特征自动区分：
    - short_circuit (短路)：细长型、长宽比大、面积较大、呈桥接状
    - micro_crack (微裂纹)：细小、面积小、形状不规则、可能呈网状/断续状

    分类特征：
    1. 面积 (area)
    2. 长宽比 (aspect_ratio)
    3. 矩形度 (rectangularity = area / bbox_area)
    4. 凸包实心度 (solidity = area / convex_area)
    5. 偏心率 (eccentricity)
    6. 骨架长度/面积比
    """

    DEFECT_TYPES = ["short_circuit", "micro_crack", "unknown"]

    def __init__(self, config=None):
        """
        Args:
            config: 分类阈值配置字典
                    支持嵌套格式 {short_circuit: {...}, micro_crack: {...}}
                    也支持扁平格式 {short_min_area: ..., ...}
        """
        if config is None:
            config = {}

        # 支持嵌套配置（如 config.yaml 中的 classification.short_circuit）
        sc_cfg = config.get("short_circuit", {})
        mc_cfg = config.get("micro_crack", {})

        # 短路分类阈值
        self.short_min_area = sc_cfg.get("min_area", config.get("short_min_area", 500))
        self.short_min_aspect_ratio = sc_cfg.get("min_aspect_ratio", config.get("short_min_aspect_ratio", 3.0))
        self.short_min_rectangularity = sc_cfg.get("min_rectangularity", config.get("short_min_rectangularity", 0.6))
        self.short_min_skeleton_area_ratio = config.get("short_min_skeleton_area_ratio", 0.15)

        # 微裂纹分类阈值
        self.crack_max_area = mc_cfg.get("max_area", config.get("crack_max_area", 800))
        self.crack_max_solidity = mc_cfg.get("min_solidity", config.get("crack_max_solidity", 0.7))
        self.crack_min_eccentricity = mc_cfg.get("min_eccentricity", config.get("crack_min_eccentricity", 0.85))
        self.crack_min_skeleton_area_ratio = mc_cfg.get("min_skeleton_area_ratio", config.get("crack_min_skeleton_area_ratio", 0.15))

        # 通用分类配置
        self.min_area = config.get("min_area", 20)
        self.score_threshold = config.get("score_threshold", 0.5)

    def extract_features(self, binary_mask, label_id, labels, stats, centroids):
        """
        从单个连通区域提取形态学特征

        Args:
            binary_mask: 二值掩码 [H, W]
            label_id: 连通区域标签 ID
            labels: 连通区域标签图 [H, W]
            stats: cv2.connectedComponentsWithStats 的 stats 输出
            centroids: cv2.connectedComponentsWithStats 的 centroids 输出

        Returns:
            DefectRegion: 缺陷区域特征对象
        """
        # 提取单个缺陷区域
        region_mask = (labels == label_id).astype(np.uint8)

        # 基本特征
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = int(centroids[label_id][0]), int(centroids[label_id][1])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        bbox = (x, y, w, h)

        # 长宽比（长边/短边）
        aspect_ratio = max(w, h) / max(min(w, h), 1)

        # 矩形度 = 区域面积 / 外接矩形面积
        bbox_area = w * h
        rectangularity = area / max(bbox_area, 1)

        # 周长
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            perimeter = float(cv2.arcLength(contours[0], True))
            contour_points = contours[0].reshape(-1, 2).tolist()
        else:
            perimeter = 0.0
            contour_points = []

        # 凸包实心度
        solidity = 1.0
        eccentricity = 0.0
        if contours and len(contours[0]) >= 3:
            try:
                hull = cv2.convexHull(contours[0])
                convex_area = float(cv2.contourArea(hull))
                solidity = area / max(convex_area, 1)

                # 偏心率（基于椭圆拟合）
                if len(contours[0]) >= 5:
                    ellipse = cv2.fitEllipse(contours[0])
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    eccentricity = np.sqrt(1 - (minor_axis / max(major_axis, 1)) ** 2)
            except Exception:
                pass

        # 骨架长度估算（细化 + 像素计数）
        skeleton = self._skeletonize(region_mask)
        skeleton_length = float(np.sum(skeleton > 0))

        return DefectRegion(
            label_id=label_id,
            area=area,
            centroid=(cx, cy),
            bbox=bbox,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            perimeter=perimeter,
            rectangularity=rectangularity,
            solidity=solidity,
            eccentricity=eccentricity,
            skeleton_length=skeleton_length,
            contour_points=contour_points,
        )

    def _skeletonize(self, binary_mask):
        """
        形态学细化（骨架提取）

        Args:
            binary_mask: 二值掩码 [H, W]

        Returns:
            numpy array: 骨架图
        """
        size = np.size(binary_mask)
        skel = np.zeros(binary_mask.shape, np.uint8)

        img = binary_mask.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open_img)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            if cv2.countNonZero(img) == 0:
                break

        return skel

    def classify(self, defect: DefectRegion) -> Tuple[str, float]:
        """
        对单个缺陷区域进行分类

        Args:
            defect: DefectRegion 特征对象

        Returns:
            tuple: (classification, confidence)
        """
        area = defect.area
        aspect_ratio = defect.aspect_ratio
        rectangularity = defect.rectangularity
        solidity = defect.solidity
        eccentricity = defect.eccentricity
        skeleton_area_ratio = defect.skeleton_length / max(area, 1)

        # 短路得分（越高越可能是短路）
        short_score = 0.0

        # 面积：短路通常面积较大
        if area >= self.short_min_area:
            short_score += 0.25
        short_score += min(area / max(self.short_min_area, 1), 1.0) * 0.1

        # 长宽比：短路细长
        if aspect_ratio >= self.short_min_aspect_ratio:
            short_score += 0.3
        short_score += min(aspect_ratio / max(self.short_min_aspect_ratio, 1), 1.0) * 0.1

        # 矩形度：短路呈长条矩形
        if rectangularity >= self.short_min_rectangularity:
            short_score += 0.15

        # 骨架/面积比：短路骨架占比高
        if skeleton_area_ratio >= self.short_min_skeleton_area_ratio:
            short_score += 0.1

        # 微裂纹得分
        crack_score = 0.0

        # 面积：裂纹通常面积小
        if area <= self.crack_max_area:
            crack_score += 0.3
        crack_score += max(0, 1 - area / max(self.crack_max_area, 1)) * 0.1

        # 凸包实心度：裂纹形状不规则，实心度低
        if solidity <= self.crack_max_solidity:
            crack_score += 0.25

        # 偏心率：裂纹偏心率高（细长不规则）
        if eccentricity >= self.crack_min_eccentricity:
            crack_score += 0.2
        crack_score += min(eccentricity, 1.0) * 0.05

        # 骨架/面积比：裂纹骨架占比也高
        if skeleton_area_ratio >= 0.1:
            crack_score += 0.1

        # 归一化
        short_score = min(short_score, 1.0)
        crack_score = min(crack_score, 1.0)

        # 决策
        if short_score >= 0.5 and short_score >= crack_score:
            return "short_circuit", short_score
        elif crack_score >= 0.4:
            return "micro_crack", crack_score
        else:
            return "unknown", max(short_score, crack_score)

    def classify_mask(self, binary_mask, min_area=50):
        """
        对整张掩码进行缺陷分类统计

        Args:
            binary_mask: 二值缺陷掩码 [H, W]，值为 0 或 255
            min_area: 最小缺陷面积阈值

        Returns:
            tuple: (defects, statistics)
                defects: List[DefectRegion]，分类后的缺陷列表
                statistics: dict，分类统计结果
        """
        # 连通区域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        defects = []

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            # 提取特征
            defect = self.extract_features(binary_mask, i, labels, stats, centroids)

            # 分类
            classification, confidence = self.classify(defect)
            defect.classification = classification
            defect.confidence = confidence

            defects.append(defect)

        # 统计
        statistics = self._compute_statistics(defects, binary_mask.shape)

        return defects, statistics

    def _compute_statistics(self, defects, image_shape):
        """
        计算分类统计结果

        Args:
            defects: 缺陷列表
            image_shape: 图像形状 (H, W)

        Returns:
            dict: 结构化统计结果
        """
        h, w = image_shape[:2]
        total_pixels = h * w

        stats = {
            "total_defects": len(defects),
            "total_defect_area": 0,
            "total_defect_ratio": 0.0,
            "by_type": {},
        }

        # 按类型统计
        for defect_type in self.DEFECT_TYPES:
            type_defects = [d for d in defects if d.classification == defect_type]
            type_area = sum(d.area for d in type_defects)

            stats["by_type"][defect_type] = {
                "count": len(type_defects),
                "total_area": int(type_area),
                "area_ratio": type_area / max(total_pixels, 1),
                "avg_area": type_area / max(len(type_defects), 1),
                "defects": [
                    {
                        "area": d.area,
                        "centroid": {"x": d.centroid[0], "y": d.centroid[1]},
                        "bbox": {
                            "x": d.bbox[0], "y": d.bbox[1],
                            "w": d.bbox[2], "h": d.bbox[3]
                        },
                        "confidence": float(d.confidence),
                    }
                    for d in type_defects
                ],
            }

        stats["total_defect_area"] = int(sum(d.area for d in defects))
        stats["total_defect_ratio"] = stats["total_defect_area"] / max(total_pixels, 1)

        return stats


class DefectReportGenerator:
    """
    PCB 缺陷检测结构化报表生成器
    """

    @staticmethod
    def generate_single_report(image_path, image_size, class_score,
                                 has_defect, statistics, defects):
        """
        生成单张 PCB 的结构化检测报表

        Args:
            image_path: 图像路径
            image_size: 图像尺寸 {"width": w, "height": h}
            class_score: 图像级分类分数
            has_defect: 是否有缺陷
            statistics: 分类统计结果
            defects: 缺陷列表

        Returns:
            dict: 结构化报表
        """
        report = {
            "image": {
                "path": str(image_path),
                "width": image_size["width"],
                "height": image_size["height"],
            },
            "classification": {
                "has_defect": bool(has_defect),
                "confidence": float(class_score),
            },
            "summary": {
                "total_defects": statistics["total_defects"],
                "total_defect_area": statistics["total_defect_area"],
                "total_defect_ratio": statistics["total_defect_ratio"],
            },
            "defect_types": {},
        }

        for defect_type, type_stats in statistics["by_type"].items():
            if type_stats["count"] > 0:
                report["defect_types"][defect_type] = {
                    "count": type_stats["count"],
                    "total_area": type_stats["total_area"],
                    "area_ratio": type_stats["area_ratio"],
                    "avg_area": type_stats["avg_area"],
                }

        return report

    @staticmethod
    def generate_batch_report(results, output_path=None):
        """
        生成批量 PCB 的汇总统计报表

        Args:
            results: 单图推理结果列表
            output_path: 报表保存路径（可选）

        Returns:
            dict: 批量汇总报表
        """
        total_images = len(results)
        defect_images = sum(1 for r in results if r.get("has_defect", False))

        batch_stats = {
            "total_images": total_images,
            "defect_images": defect_images,
            "normal_images": total_images - defect_images,
            "defect_image_ratio": defect_images / max(total_images, 1),
            "by_type_summary": {
                "short_circuit": {"count": 0, "total_area": 0, "image_count": 0},
                "micro_crack": {"count": 0, "total_area": 0, "image_count": 0},
                "unknown": {"count": 0, "total_area": 0, "image_count": 0},
            },
            "per_image_reports": [],
        }

        for result in results:
            classification = result.get("classification_report", {})
            defect_types = classification.get("defect_types", {})

            per_image = {
                "image_path": result.get("image_path", ""),
                "has_defect": result.get("has_defect", False),
                "confidence": result.get("class_score", 0.0),
                "total_defects": classification.get("summary", {}).get("total_defects", 0),
                "total_defect_area": classification.get("summary", {}).get("total_defect_area", 0),
                "defect_types": list(defect_types.keys()),
            }
            batch_stats["per_image_reports"].append(per_image)

            # 按类型汇总
            for dt, ds in defect_types.items():
                if dt in batch_stats["by_type_summary"]:
                    batch_stats["by_type_summary"][dt]["count"] += ds.get("count", 0)
                    batch_stats["by_type_summary"][dt]["total_area"] += ds.get("total_area", 0)
                    if ds.get("count", 0) > 0:
                        batch_stats["by_type_summary"][dt]["image_count"] += 1

        if output_path is not None:
            import json
            from pathlib import Path
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(batch_stats, f, indent=2, ensure_ascii=False)

        return batch_stats

    @staticmethod
    def format_report_text(report):
        """
        将报表格式化为可读文本

        Args:
            report: 批量汇总报表字典

        Returns:
            str: 格式化文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PCB 缺陷检测批量统计报表")
        lines.append("=" * 60)
        lines.append(f"总图片数:       {report['total_images']}")
        lines.append(f"缺陷图片数:     {report['defect_images']}")
        lines.append(f"正常图片数:     {report['normal_images']}")
        lines.append(f"缺陷图片占比:   {report['defect_image_ratio']:.2%}")
        lines.append("")
        lines.append("--- 缺陷类型汇总 ---")

        type_names = {
            "short_circuit": "短路 (Short Circuit)",
            "micro_crack": "微裂纹 (Micro Crack)",
            "unknown": "未知类型 (Unknown)",
        }

        for dt, ds in report["by_type_summary"].items():
            name = type_names.get(dt, dt)
            lines.append(f"  {name}:")
            lines.append(f"    缺陷总数:   {ds['count']}")
            lines.append(f"    出现图片数: {ds['image_count']}")
            lines.append(f"    总面积:     {ds['total_area']} 像素")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
