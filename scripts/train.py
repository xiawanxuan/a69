
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from utils import load_config, get_device, get_device_info, set_seed, save_config
from models import build_model
from losses import build_loss
from data import build_dataloader


class Trainer:
    """
    PCB 弱监督分割训练器
    """

    def __init__(self, config, split_file):
        self.config = config
        self.device = get_device(config)
        self.split_file = split_file

        # 设置随机种子
        seed = config.get("train", {}).get("seed", 42)
        set_seed(seed)

        # 构建模型
        self.model = build_model(config)
        self.model = self.model.to(self.device)

        # 构建损失函数
        self.criterion = build_loss(config)

        # 构建优化器
        train_cfg = config.get("train", {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0001),
        )

        # 学习率调度器
        self.epochs = train_cfg.get("epochs", 100)
        warmup_epochs = train_cfg.get("warmup_epochs", 5)
        lr_scheduler_type = train_cfg.get("lr_scheduler", "cosine")

        if lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - warmup_epochs
            )
        else:
            main_scheduler = None

        # Warmup 调度器
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs

        # 梯度裁剪
        self.gradient_clip = train_cfg.get("gradient_clip", None)

        # 数据加载器
        self.train_loader = build_dataloader(config, split_file=split_file, mode="train")
        self.val_loader = build_dataloader(config, split_file=split_file, mode="val")

        # 检查点和日志目录
        paths_cfg = config.get("paths", {})
        self.checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 最佳模型追踪
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # 打印信息
        device_info = get_device_info(self.device)
        print(f"设备信息: {device_info}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    def train_epoch(self, epoch):
        """
        训练一个 epoch
        """
        self.model.train()

        total_loss = 0.0
        loss_details = {}
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            image_labels = batch["image_label"].to(self.device)

            # 是否有像素级标签
            pixel_mask = None
            has_pixel = "pixel_mask" in batch
            if has_pixel:
                pixel_mask = batch["pixel_mask"].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 计算损失
            loss_dict = self.criterion(
                outputs, image_labels,
                with_pixel_mask=has_pixel,
                pixel_mask=pixel_mask,
            )

            loss = loss_dict["total_loss"]

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_details:
                    loss_details[k] = 0.0
                loss_details[k] += v.item()

            num_batches += 1

            # 打印进度
            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        for k in loss_details:
            loss_details[k] /= num_batches

        elapsed = time.time() - start_time

        return avg_loss, loss_details, elapsed

    @torch.no_grad()
    def validate(self):
        """
        验证
        """
        self.model.eval()

        total_loss = 0.0
        loss_details = {}
        num_batches = 0

        # 分类准确率
        correct = 0
        total = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            image_labels = batch["image_label"].to(self.device)

            pixel_mask = None
            has_pixel = "pixel_mask" in batch
            if has_pixel:
                pixel_mask = batch["pixel_mask"].to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(
                outputs, image_labels,
                with_pixel_mask=has_pixel,
                pixel_mask=pixel_mask,
            )

            total_loss += loss_dict["total_loss"].item()
            for k, v in loss_dict.items():
                if k not in loss_details:
                    loss_details[k] = 0.0
                loss_details[k] += v.item()

            num_batches += 1

            # 分类准确率
            class_pred = (torch.sigmoid(outputs["class_logit"]) > 0.5).long().view(-1)
            correct += (class_pred == image_labels).sum().item()
            total += image_labels.size(0)

        avg_loss = total_loss / num_batches
        for k in loss_details:
            loss_details[k] /= num_batches

        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, loss_details, accuracy

    def save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)

        # 按周期保存
        if (epoch + 1) % 10 == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch + 1}.pth"
            torch.save(checkpoint, epoch_path)

    def train(self):
        """
        训练主循环
        """
        print(f"\n开始训练，共 {self.epochs} 个 epoch\n")

        for epoch in range(self.epochs):
            # 训练
            train_loss, train_details, train_time = self.train_epoch(epoch)

            # 学习率调度
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            elif self.main_scheduler is not None:
                self.main_scheduler.step()

            # 验证
            val_loss, val_details, val_acc = self.validate()

            # 判断是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1

            # 保存检查点
            self.save_checkpoint(epoch, is_best=is_best)

            # 打印日志
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n{'=' * 60}")
            print(f"Epoch [{epoch + 1}/{self.epochs}] - 用时: {train_time:.1f}s")
            print(f"  训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            print(f"  验证分类准确率: {val_acc:.4f}")
            print(f"  学习率: {lr:.6f}")
            print(f"  最佳验证损失: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
            print(f"{'=' * 60}\n")

        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f} (epoch {self.best_epoch})")


def main():
    parser = argparse.ArgumentParser(description="PCB 弱监督分割训练脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--split_file", type=str, default="datasets/dataset_split.json",
                        help="数据集划分文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 创建训练器
    trainer = Trainer(config, args.split_file)

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"从 {args.resume} 恢复训练")

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
