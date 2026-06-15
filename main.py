
#!/usr/bin/env python3
"""
项目入口脚本

使用示例：
    # 数据集划分
    python main.py split --image_dir datasets/images --output datasets

    # 训练
    python main.py train --config configs/config.yaml

    # 评估
    python main.py evaluate --checkpoint checkpoints/best.pth

    # 批量推理
    python main.py predict --checkpoint checkpoints/best.pth --input datasets/test

    # 启动 API 服务
    python main.py serve --checkpoint checkpoints/best.pth --port 8000
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_split(args):
    """数据集划分"""
    from data.split_dataset import split_dataset

    split_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output,
    )


def cmd_train(args):
    """训练"""
    from scripts.train import Trainer
    from utils import load_config

    config = load_config(args.config)
    trainer = Trainer(config, args.split_file)

    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"从 {args.resume} 恢复训练")

    trainer.train()


def cmd_evaluate(args):
    """评估"""
    from scripts.evaluate import evaluate_checkpoint
    import json

    results = evaluate_checkpoint(
        args.checkpoint,
        args.split_file,
        args.config,
        args.mode,
    )

    print(f"\n评估结果 ({args.mode} 集):")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


def cmd_predict(args):
    """推理"""
    from inference import PCBSegmentor
    from pathlib import Path
    from PIL import Image

    segmentor = PCBSegmentor(args.checkpoint, args.config, enable_classification=not args.no_classification)

    if args.threshold is not None:
        segmentor.threshold = args.threshold
    if args.min_area is not None:
        segmentor.min_defect_area = args.min_area

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("outputs/inference")

    if input_path.is_dir():
        segmentor.predict_folder(
            input_path,
            output_dir=str(output_dir),
            save_mask=not args.no_mask,
            save_overlay=not args.no_overlay,
            save_report=not args.no_classification and not args.no_report,
        )
    else:
        result = segmentor.predict(str(input_path))
        print(f"\n推理结果:")
        print(f"  有缺陷: {result['has_defect']}")
        print(f"  分类分数: {result['class_score']:.4f}")
        print(f"  缺陷面积: {result['defect_area']} 像素")
        print(f"  缺陷占比: {result['defect_ratio']:.2%}")

        # 打印分类统计
        if result.get("classification_report"):
            cr = result["classification_report"]
            print(f"\n  缺陷分类统计:")
            type_names = {
                "short_circuit": "    短路 (Short Circuit)",
                "micro_crack": "    微裂纹 (Micro Crack)",
                "unknown": "    未知类型 (Unknown)",
            }
            for dt, ds in cr.get("defect_types", {}).items():
                name = type_names.get(dt, f"    {dt}")
                print(f"{name}: {ds['count']} 个, 总面积 {ds['total_area']} 像素")

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"{input_path.stem}_mask.png"
        Image.fromarray(result["mask"]).save(mask_path)
        print(f"  掩码已保存: {mask_path}")


def cmd_serve(args):
    """启动 API 服务"""
    import uvicorn
    from api.fastapi_app import app, segmentor as api_segmentor

    if args.checkpoint:
        from inference import PCBSegmentor
        api_segmentor.__class__  # 确保导入
        # 这里通过环境变量等方式传参，实际由 fastapi 启动事件处理

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="PCB 弱监督缺陷分割 - 项目入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 数据集划分
    split_parser = subparsers.add_parser("split", help="划分数据集")
    split_parser.add_argument("--image_dir", type=str, required=True, help="图像目录")
    split_parser.add_argument("--label_dir", type=str, default=None, help="标签目录")
    split_parser.add_argument("--train_ratio", type=float, default=0.8)
    split_parser.add_argument("--val_ratio", type=float, default=0.1)
    split_parser.add_argument("--test_ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--output", type=str, default="datasets")

    # 训练
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, default="configs/config.yaml")
    train_parser.add_argument("--split_file", type=str, default="datasets/dataset_split.json")
    train_parser.add_argument("--resume", type=str, default=None)

    # 评估
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--split_file", type=str, default="datasets/dataset_split.json")
    eval_parser.add_argument("--config", type=str, default=None)
    eval_parser.add_argument("--mode", type=str, default="test", choices=["train", "val", "test"])
    eval_parser.add_argument("--output", type=str, default=None)

    # 推理
    pred_parser = subparsers.add_parser("predict", help="模型推理")
    pred_parser.add_argument("--checkpoint", type=str, required=True)
    pred_parser.add_argument("--input", type=str, required=True, help="输入图片或文件夹")
    pred_parser.add_argument("--config", type=str, default=None)
    pred_parser.add_argument("--output", type=str, default=None)
    pred_parser.add_argument("--threshold", type=float, default=None)
    pred_parser.add_argument("--min_area", type=int, default=None)
    pred_parser.add_argument("--no_mask", action="store_true")
    pred_parser.add_argument("--no_overlay", action="store_true")
    pred_parser.add_argument("--no_classification", action="store_true",
                             help="禁用缺陷自动分类统计子模块")
    pred_parser.add_argument("--no_report", action="store_true",
                             help="不保存结构化统计报表")

    # API 服务
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务")
    serve_parser.add_argument("--checkpoint", type=str, default=None)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "split":
        cmd_split(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
