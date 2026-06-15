
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms


def mirror_pad_resize(image, target_size, pad_value=0, return_pad_info=False):
    """
    边缘镜像填充 + 等比例缩放

    先将图像等比例缩放到能放入目标尺寸的最大大小，
    然后用边缘镜像填充到目标尺寸，避免直接拉伸缩放导致边缘缺陷变形。

    Args:
        image: 输入图像 (PIL Image, numpy array [H, W, C] 或 [H, W])
        target_size: 目标尺寸 (height, width)
        pad_value: 当不需要镜像时的填充值（默认 0，黑色）
        return_pad_info: 是否返回填充信息，用于后续裁剪

    Returns:
        如果 return_pad_info=False:
            resized_padded: 处理后的图像 (numpy array)
        如果 return_pad_info=True:
            tuple: (resized_padded, pad_info)
            pad_info: dict，包含缩放和填充信息，用于还原
    """
    target_h, target_w = target_size

    # 统一转为 numpy array
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = image.copy()

    orig_h, orig_w = img.shape[:2]

    # 计算等比例缩放因子
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # 等比例缩放
    if new_h != orig_h or new_w != orig_w:
        if img.ndim == 2 or img.shape[2] == 1:
            interpolation = cv2.INTER_NEAREST  # 掩码用最近邻
        else:
            interpolation = cv2.INTER_LINEAR  # 图像用双线性
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # 计算上下左右需要填充的像素数
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    # 边缘镜像填充
    if img.ndim == 2:
        # 单通道（掩码）
        padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
    else:
        # 多通道（彩色图）
        padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )

    if not return_pad_info:
        return padded

    pad_info = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "target_h": target_h,
        "target_w": target_w,
    }

    return padded, pad_info


def unpad_image(padded_image, pad_info, return_orig_size=True):
    """
    去除填充区域，裁剪回缩放后的尺寸（可选：再缩放回原始尺寸）

    Args:
        padded_image: 填充后的图像 (numpy array [H, W, C] 或 [H, W])
        pad_info: mirror_pad_resize 返回的填充信息字典
        return_orig_size: 是否缩放到原始尺寸（True=原始尺寸, False=缩放后尺寸）

    Returns:
        numpy array: 裁剪/缩放后的图像
    """
    pad_top = pad_info["pad_top"]
    pad_bottom = pad_info["pad_bottom"]
    pad_left = pad_info["pad_left"]
    pad_right = pad_info["pad_right"]

    target_h = pad_info["target_h"]
    target_w = pad_info["target_w"]

    # 裁剪掉填充区域
    cropped = padded_image[
        pad_top:target_h - pad_bottom,
        pad_left:target_w - pad_right
    ]

    if return_orig_size:
        orig_h = pad_info["orig_h"]
        orig_w = pad_info["orig_w"]
        if cropped.ndim == 2 or cropped.shape[2] == 1:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
        cropped = cv2.resize(cropped, (orig_w, orig_h), interpolation=interpolation)

    return cropped


def unpad_points(points, pad_info):
    """
    将填充后图像上的坐标点转换回原始图像坐标

    Args:
        points: 坐标点列表 [{"x": x, "y": y, ...}, ...]
        pad_info: 填充信息字典

    Returns:
        list: 转换后的坐标点列表
    """
    pad_left = pad_info["pad_left"]
    pad_top = pad_info["pad_top"]
    scale = pad_info["scale"]

    converted = []
    for pt in points:
        new_pt = pt.copy()
        # 先减去填充偏移，再除以缩放比例
        new_pt["x"] = int(round((pt["x"] - pad_left) / scale))
        new_pt["y"] = int(round((pt["y"] - pad_top) / scale))
        # 面积也要按比例还原
        if "area" in pt:
            new_pt["area"] = int(round(pt["area"] / (scale * scale)))
        converted.append(new_pt)

    return converted


class MirrorPadResize(object):
    """
    可组合进 transforms 的镜像填充缩放类

    使用方式：
        transform = transforms.Compose([
            MirrorPadResize(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(...)
        ])
    """

    def __init__(self, size, is_mask=False):
        """
        Args:
            size: 目标尺寸 (height, width) 或 int（正方形）
            is_mask: 是否是掩码（影响插值方式）
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.is_mask = is_mask
        self._last_pad_info = None

    def __call__(self, image):
        """
        Args:
            image: PIL Image

        Returns:
            PIL Image: 处理后的图像
        """
        img_array = np.array(image)

        if self.is_mask:
            # 掩码用最近邻
            padded, pad_info = mirror_pad_resize(img_array, self.size, return_pad_info=True)
        else:
            padded, pad_info = mirror_pad_resize(img_array, self.size, return_pad_info=True)

        self._last_pad_info = pad_info

        if padded.ndim == 2:
            return Image.fromarray(padded, mode="L")
        else:
            return Image.fromarray(padded, mode="RGB")

    def get_last_pad_info(self):
        """获取上一次调用的填充信息"""
        return self._last_pad_info


def build_transform_with_padding(image_size, augmentation_config=None, mode="train"):
    """
    构建带镜像填充的数据增强流水线

    Args:
        image_size: 目标图像尺寸 (height, width)
        augmentation_config: 增强配置字典
        mode: "train" 或 "val"/"test"

    Returns:
        transforms.Compose: 增强流水线
        MirrorPadResize: 镜像填充变换实例（用于获取 pad_info）
    """
    pipeline_list = []

    # 训练时的几何和光度增强（作用于原始尺寸的 PIL Image）
    if mode == "train" and augmentation_config is not None:
        geo_cfg = augmentation_config.get("geometric", {})
        photo_cfg = augmentation_config.get("photometric", {})

        if geo_cfg.get("horizontal_flip_prob", 0) > 0:
            pipeline_list.append(
                transforms.RandomHorizontalFlip(p=geo_cfg["horizontal_flip_prob"])
            )

        if geo_cfg.get("vertical_flip_prob", 0) > 0:
            pipeline_list.append(
                transforms.RandomVerticalFlip(p=geo_cfg["vertical_flip_prob"])
            )

        if photo_cfg.get("brightness_prob", 0) > 0 or photo_cfg.get("contrast_prob", 0) > 0:
            brightness = tuple(photo_cfg.get("brightness_range", [0.8, 1.2])) if photo_cfg.get("brightness_prob", 0) > 0 else 0
            contrast = tuple(photo_cfg.get("contrast_range", [0.8, 1.2])) if photo_cfg.get("contrast_prob", 0) > 0 else 0
            pipeline_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast)
            )

    # 镜像填充 + 缩放
    mirror_pad = MirrorPadResize(size=image_size)
    pipeline_list.append(mirror_pad)

    # 转 Tensor
    pipeline_list.append(transforms.ToTensor())

    # 噪声增强（作用于 Tensor）
    if mode == "train" and augmentation_config is not None:
        noise_cfg = augmentation_config.get("noise", {})
        if (noise_cfg.get("gaussian_prob", 0) > 0 or
            noise_cfg.get("salt_pepper_prob", 0) > 0 or
            noise_cfg.get("speckle_prob", 0) > 0 or
            noise_cfg.get("poisson_prob", 0) > 0):
            from augmentations import NoiseCompose
            noise_transform = NoiseCompose.from_config(augmentation_config)
            if noise_transform.noise_types:
                pipeline_list.append(noise_transform)

    # 标准化
    pipeline_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return transforms.Compose(pipeline_list), mirror_pad


def build_mask_transform_with_padding(image_size):
    """
    构建掩码的镜像填充变换

    Args:
        image_size: 目标图像尺寸 (height, width)

    Returns:
        tuple: (transforms.Compose, MirrorPadResize)
    """
    mirror_pad = MirrorPadResize(size=image_size, is_mask=True)

    transform = transforms.Compose([
        mirror_pad,
        transforms.ToTensor(),
    ])

    return transform, mirror_pad
