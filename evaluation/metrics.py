
import numpy as np
import torch
from typing import Dict, List


def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """
    计算交并比 (IoU / Jaccard Index)

    Args:
        pred_mask: 预测掩码，可以是概率图或二值图 [H, W] 或 [B, 1, H, W]
        gt_mask: 真实掩码 [H, W] 或 [B, 1, H, W]
        threshold: 二值化阈值

    Returns:
        float: IoU 值
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    # 二值化
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0.5).astype(np.float32)

    # 计算交集和并集
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection

    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0

    return intersection / union


def compute_dice(pred_mask, gt_mask, threshold=0.5):
    """
    计算 Dice 系数 (F1 score for segmentation)

    Args:
        pred_mask: 预测掩码
        gt_mask: 真实掩码
        threshold: 二值化阈值

    Returns:
        float: Dice 系数
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0.5).astype(np.float32)

    intersection = np.sum(pred_binary * gt_binary)
    total = np.sum(pred_binary) + np.sum(gt_binary)

    if total == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0

    return 2 * intersection / total


def compute_precision(pred_mask, gt_mask, threshold=0.5):
    """
    计算精确率 (Precision)

    Args:
        pred_mask: 预测掩码
        gt_mask: 真实掩码
        threshold: 二值化阈值

    Returns:
        float: 精确率
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0.5).astype(np.float32)

    tp = np.sum(pred_binary * gt_binary)
    fp = np.sum(pred_binary * (1 - gt_binary))

    if tp + fp == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0

    return tp / (tp + fp)


def compute_recall(pred_mask, gt_mask, threshold=0.5):
    """
    计算召回率 (Recall / Sensitivity)

    Args:
        pred_mask: 预测掩码
        gt_mask: 真实掩码
        threshold: 二值化阈值

    Returns:
        float: 召回率
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0.5).astype(np.float32)

    tp = np.sum(pred_binary * gt_binary)
    fn = np.sum((1 - pred_binary) * gt_binary)

    if tp + fn == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0

    return tp / (tp + fn)


def compute_f1(pred_mask, gt_mask, threshold=0.5):
    """
    计算 F1 分数

    Args:
        pred_mask: 预测掩码
        gt_mask: 真实掩码
        threshold: 二值化阈值

    Returns:
        float: F1 分数
    """
    precision = compute_precision(pred_mask, gt_mask, threshold)
    recall = compute_recall(pred_mask, gt_mask, threshold)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_image_level_accuracy(pred_labels, gt_labels):
    """
    计算图像级分类准确率

    Args:
        pred_labels: 预测标签列表或数组
        gt_labels: 真实标签列表或数组

    Returns:
        float: 准确率
    """
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()

    pred_labels = np.array(pred_labels).flatten()
    gt_labels = np.array(gt_labels).flatten()

    correct = np.sum(pred_labels == gt_labels)
    total = len(gt_labels)

    return correct / total if total > 0 else 0.0


class SegmentationMetrics:
    """
    分割评估指标计算器

    支持计算：
    - IoU (交并比)
    - Dice (F1)
    - Precision (精确率)
    - Recall (召回率)
    - F1 Score
    - 图像级分类准确率
    """

    def __init__(self, metrics=None, threshold=0.5):
        """
        Args:
            metrics: 要计算的指标列表，默认 ["iou", "dice", "precision", "recall", "f1"]
            threshold: 二值化阈值
        """
        if metrics is None:
            metrics = ["iou", "dice", "precision", "recall", "f1"]
        self.metrics = metrics
        self.threshold = threshold

        # 存储每个样本的指标
        self.per_image_metrics = {m: [] for m in metrics}
        self.image_preds = []
        self.image_labels = []

        self.metric_functions = {
            "iou": compute_iou,
            "dice": compute_dice,
            "precision": compute_precision,
            "recall": compute_recall,
            "f1": compute_f1,
        }

    def update(self, pred_mask, gt_mask, pred_label=None, gt_label=None):
        """
        更新指标

        Args:
            pred_mask: 预测掩码 [H, W] 或 [B, 1, H, W]
            gt_mask: 真实掩码 [H, W] 或 [B, 1, H, W]
            pred_label: 预测的图像级标签
            gt_label: 真实的图像级标签
        """
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy()

        # 处理 batch
        if pred_mask.ndim == 4:
            batch_size = pred_mask.shape[0]
            for i in range(batch_size):
                self._update_single(
                    pred_mask[i, 0], gt_mask[i, 0] if gt_mask.ndim == 4 else gt_mask[i]
                )
        else:
            self._update_single(pred_mask, gt_mask)

        # 图像级标签
        if pred_label is not None and gt_label is not None:
            if isinstance(pred_label, torch.Tensor):
                pred_label = pred_label.detach().cpu().numpy().flatten().tolist()
            if isinstance(gt_label, torch.Tensor):
                gt_label = gt_label.detach().cpu().numpy().flatten().tolist()
            if not isinstance(pred_label, list):
                pred_label = [pred_label]
            if not isinstance(gt_label, list):
                gt_label = [gt_label]

            self.image_preds.extend(pred_label)
            self.image_labels.extend(gt_label)

    def _update_single(self, pred_mask, gt_mask):
        """
        更新单张图片的指标
        """
        for metric_name in self.metrics:
            if metric_name in self.metric_functions:
                value = self.metric_functions[metric_name](
                    pred_mask, gt_mask, self.threshold
                )
                self.per_image_metrics[metric_name].append(value)

    def compute(self):
        """
        计算所有指标的平均值

        Returns:
            dict: 各指标的平均值
        """
        result = {}

        for metric_name in self.metrics:
            values = self.per_image_metrics[metric_name]
            if values:
                result[metric_name] = float(np.mean(values))
                result[f"{metric_name}_std"] = float(np.std(values))
            else:
                result[metric_name] = 0.0
                result[f"{metric_name}_std"] = 0.0

        # 图像级准确率
        if self.image_labels and self.image_preds:
            result["image_accuracy"] = compute_image_level_accuracy(
                self.image_preds, self.image_labels
            )

        result["num_samples"] = len(self.per_image_metrics.get("iou", []))

        return result

    def reset(self):
        """
        重置所有指标
        """
        self.per_image_metrics = {m: [] for m in self.metrics}
        self.image_preds = []
        self.image_labels = []


def evaluate_model(model, dataloader, device, metrics=None, threshold=0.5):
    """
    评估模型在数据集上的表现

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备
        metrics: 指标列表
        threshold: 二值化阈值

    Returns:
        dict: 评估结果
    """
    model.eval()
    metric_calculator = SegmentationMetrics(metrics=metrics, threshold=threshold)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            image_labels = batch["image_label"].to(device)

            outputs = model(images)
            mask_prob = torch.sigmoid(outputs["mask"])
            class_pred = (torch.sigmoid(outputs["class_logit"]) > threshold).long()

            # 如果有像素级标签
            if "pixel_mask" in batch:
                pixel_masks = batch["pixel_mask"].to(device)
                metric_calculator.update(
                    mask_prob, pixel_masks,
                    pred_label=class_pred,
                    gt_label=image_labels
                )
            else:
                # 只有图像级标签时，只计算图像级准确率
                metric_calculator.image_preds.extend(
                    class_pred.cpu().numpy().flatten().tolist()
                )
                metric_calculator.image_labels.extend(
                    image_labels.cpu().numpy().flatten().tolist()
                )

    return metric_calculator.compute()
