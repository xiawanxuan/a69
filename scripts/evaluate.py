
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np

from utils import load_config, get_device
from models import build_model
from data import build_dataloader
from evaluation import evaluate_model, SegmentationMetrics


def evaluate_checkpoint(checkpoint_path, split_file, config_path=None, mode="test"):
    """
    评估模型检查点

    Args:
        checkpoint_path: 检查点路径
        split_file: 数据集划分文件
        config_path: 配置文件路径
        mode: 评估模式 "val" 或 "test"

    Returns:
        dict: 评估结果
    """
    # 加载检查点
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 加载配置
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = checkpoint.get("config", {})

    device = get_device(config)

    # 构建模型
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 构建数据加载器
    dataloader = build_dataloader(config, split_file=split_file, mode=mode)

    # 评估指标
    eval_cfg = config.get("evaluation", {})
    metrics = eval_cfg.get("metrics", ["iou", "dice", "precision", "recall", "f1"])
    threshold = config.get("inference", {}).get("threshold", 0.5)

    # 执行评估
    results = evaluate_model(model, dataloader, device, metrics=metrics, threshold=threshold)

    return results


def main():
    parser = argparse.ArgumentParser(description="PCB 分割模型评估脚本")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--split_file", type=str,
                        default="datasets/dataset_split.json",
                        help="数据集划分文件")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="评估模式")
    parser.add_argument("--output", type=str, default="outputs/evaluation",
                        help="输出目录")
    parser.add_argument("--threshold", type=float, default=None,
                        help="二值化阈值")

    args = parser.parse_args()

    # 执行评估
    results = evaluate_checkpoint(
        args.checkpoint,
        args.split_file,
        args.config,
        args.mode,
    )

    # 覆盖阈值参数
    if args.threshold is not None:
        results["threshold"] = args.threshold

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{args.mode}_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"评估结果 ({args.mode} 集)")
    print(f"{'=' * 60}")
    print(f"  样本数量: {results.get('num_samples', 0)}")
    for metric, value in results.items():
        if isinstance(value, float) and not metric.endswith("_std"):
            std = results.get(f"{metric}_std", 0)
            print(f"  {metric}: {value:.4f} ± {std:.4f}")
        elif metric == "image_accuracy":
            print(f"  {metric}: {value:.4f}")
    print(f"{'=' * 60}")
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
