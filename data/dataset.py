
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import json


class PCBWeakDataset(Dataset):
    """
    PCB 弱监督分割数据集

    支持两种模式：
    1. 完全弱监督：只有图像级标签（有/无缺陷）
    2. 半监督：部分图像有像素级标签

    Args:
        samples: 样本列表，每个元素是字典 {image, label, image_label}
        transform: 图像变换
        image_size: 图像尺寸 [height, width]
        return_pixel_label: 是否返回像素级标签
        use_mirror_pad: 是否使用边缘镜像填充（替代直接 Resize）
    """

    def __init__(self, samples, transform=None, image_size=(256, 256),
                 return_pixel_label=False, use_mirror_pad=False):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size
        self.return_pixel_label = return_pixel_label
        self.use_mirror_pad = use_mirror_pad

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample["image"]).convert("RGB")

        # 加载像素级标签（如果有）
        pixel_mask = None
        if self.return_pixel_label and sample.get("label"):
            pixel_mask = Image.open(sample["label"]).convert("L")

        # 应用变换
        pad_info = None
        if self.transform is not None:
            # 如果有像素掩码，需要同步变换
            if pixel_mask is not None:
                # 使用相同的随机种子保证变换一致
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                image = self.transform(image)
                torch.manual_seed(seed)
                pixel_mask, pad_info = self.transform_mask(pixel_mask, return_pad_info=True)
            else:
                if self.use_mirror_pad and hasattr(self.transform, 'transforms'):
                    from utils.image_utils import mirror_pad_resize
                    # 对于镜像填充模式：先做其他增强，再镜像填充
                    image = self.transform(image)
                else:
                    image = self.transform(image)
        else:
            # 没有 transform 也做镜像填充
            if self.use_mirror_pad:
                from utils.image_utils import mirror_pad_resize
                img_array = np.array(image)
                padded, pad_info = mirror_pad_resize(img_array, self.image_size, return_pad_info=True)
                image = Image.fromarray(padded)

        # 图像级标签
        image_label = torch.tensor(sample["image_label"], dtype=torch.long)

        result = {
            "image": image,
            "image_label": image_label,
            "image_path": sample["image"],
        }

        if pixel_mask is not None:
            result["pixel_mask"] = pixel_mask

        if pad_info is not None:
            result["pad_info"] = pad_info

        return result

    def transform_mask(self, mask, return_pad_info=False):
        """
        对掩码应用变换（调整大小+转tensor）

        Args:
            mask: PIL Image 掩码
            return_pad_info: 是否返回填充信息

        Returns:
            如果 return_pad_info=False: tensor
            如果 return_pad_info=True: (tensor, pad_info)
        """
        from torchvision import transforms

        if self.use_mirror_pad:
            from utils.image_utils import MirrorPadResize
            mirror_pad = MirrorPadResize(size=self.image_size, is_mask=True)
            mask_transform = transforms.Compose([
                mirror_pad,
                transforms.ToTensor(),
            ])
            result = mask_transform(mask)
            if return_pad_info:
                return result, mirror_pad.get_last_pad_info()
            return result
        else:
            mask_transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
            result = mask_transform(mask)
            if return_pad_info:
                return result, None
            return result


def build_dataloader(
    config,
    split_file=None,
    samples=None,
    mode="train",
    augmentation=None,
    use_mirror_pad=None,
):
    """
    构建数据加载器

    Args:
        config: 配置字典
        split_file: 数据集划分文件路径
        samples: 样本列表（与 split_file 二选一）
        mode: "train", "val", 或 "test"
        augmentation: 数据增强配置（可选，默认从 config 读取）
        use_mirror_pad: 是否使用边缘镜像填充（默认从 config 读取）

    Returns:
        DataLoader: PyTorch 数据加载器
    """
    from augmentations import build_augmentation_pipeline

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [256, 256]))

    # 是否使用镜像填充
    if use_mirror_pad is None:
        use_mirror_pad = data_cfg.get("use_mirror_pad", True)

    # 加载样本
    if samples is not None:
        pass
    elif split_file is not None:
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        samples = split_data[mode]
    else:
        raise ValueError("必须提供 split_file 或 samples")

    # 数据增强
    if augmentation is None and mode == "train":
        augmentation = config.get("augmentation", {})

    if use_mirror_pad:
        from utils.image_utils import build_transform_with_padding
        transform, mirror_pad_obj = build_transform_with_padding(
            image_size,
            augmentation_config=augmentation if mode == "train" else None,
            mode=mode,
        )
    else:
        transform = build_augmentation_pipeline(
            augmentation,
            mode=mode,
            image_size=image_size,
        )
        mirror_pad_obj = None

    # 判断是否有像素级标签
    has_pixel_labels = any(s.get("label") for s in samples)

    dataset = PCBWeakDataset(
        samples=samples,
        transform=transform,
        image_size=image_size,
        return_pixel_label=has_pixel_labels,
        use_mirror_pad=use_mirror_pad,
    )

    # 数据加载器
    batch_size = config.get("train", {}).get("batch_size", 8) if mode == "train" \
        else config.get("inference", {}).get("batch_size", 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", False),
        drop_last=(mode == "train"),
    )

    return dataloader


def build_inference_dataset(image_paths, config=None, image_size=(256, 256),
                            use_mirror_pad=None):
    """
    构建推理用数据集

    Args:
        image_paths: 图片路径列表
        config: 配置字典
        image_size: 图像尺寸
        use_mirror_pad: 是否使用边缘镜像填充

    Returns:
        DataLoader: 推理数据加载器
    """
    from torchvision import transforms
    from utils.image_utils import build_transform_with_padding

    # 是否使用镜像填充
    if use_mirror_pad is None:
        if config is not None:
            use_mirror_pad = config.get("data", {}).get("use_mirror_pad", True)
        else:
            use_mirror_pad = True

    if use_mirror_pad:
        transform, mirror_pad_obj = build_transform_with_padding(
            image_size,
            augmentation_config=None,
            mode="test",
        )
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        mirror_pad_obj = None

    samples = [
        {"image": str(p), "image_label": 0, "label": None}
        for p in image_paths
    ]

    dataset = PCBWeakDataset(
        samples=samples,
        transform=transform,
        image_size=image_size,
        return_pixel_label=False,
        use_mirror_pad=use_mirror_pad,
    )

    batch_size = 4
    if config is not None:
        batch_size = config.get("inference", {}).get("batch_size", 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return dataloader
