
import torch
import numpy as np
from torchvision import transforms
import random
from PIL import Image


class GaussianNoise(object):
    """
    高斯噪声增强

    Args:
        std: 噪声标准差范围 [min_std, max_std]
        p: 应用概率
    """

    def __init__(self, std=(0.01, 0.05), p=0.5):
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        std = random.uniform(self.std[0], self.std[1])
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * std
            return torch.clamp(img + noise, 0.0, 1.0)
        else:
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, std, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 1)
            return Image.fromarray((noisy * 255).astype(np.uint8))


class SaltPepperNoise(object):
    """
    椒盐噪声增强

    Args:
        amount: 噪声点比例
        p: 应用概率
    """

    def __init__(self, amount=0.02, p=0.3):
        self.amount = amount
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img).astype(np.float32) / 255.0

        salt = np.random.random(img_np.shape) < self.amount / 2
        pepper = np.random.random(img_np.shape) < self.amount / 2

        noisy = img_np.copy()
        noisy[salt] = 1.0
        noisy[pepper] = 0.0

        if isinstance(img, torch.Tensor):
            return torch.from_numpy(noisy)
        else:
            return Image.fromarray((noisy * 255).astype(np.uint8))


class SpeckleNoise(object):
    """
    斑点噪声（乘性噪声）增强

    Args:
        std: 噪声标准差
        p: 应用概率
    """

    def __init__(self, std=0.05, p=0.2):
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + img * noise, 0.0, 1.0)
        else:
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.std, img_array.shape)
            noisy = np.clip(img_array + img_array * noise, 0, 1)
            return Image.fromarray((noisy * 255).astype(np.uint8))


class PoissonNoise(object):
    """
    泊松噪声（模拟光子噪声）增强

    Args:
        p: 应用概率
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img).astype(np.float32) / 255.0

        vals = len(np.unique(img_np))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img_np * vals) / float(vals)
        noisy = np.clip(noisy, 0, 1)

        if isinstance(img, torch.Tensor):
            return torch.from_numpy(noisy)
        else:
            return Image.fromarray((noisy * 255).astype(np.uint8))


class RandomBlur(object):
    """
    随机模糊增强

    Args:
        kernel_sizes: 模糊核大小列表
        p: 应用概率
    """

    def __init__(self, kernel_sizes=(3, 5), p=0.2):
        self.kernel_sizes = kernel_sizes
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        kernel_size = random.choice(self.kernel_sizes)
        blur_transform = transforms.GaussianBlur(kernel_size)
        return blur_transform(img)


class NoiseCompose(object):
    """
    多种噪声组合增强

    Args:
        noise_types: 噪声类型列表
        shuffle: 是否随机打乱噪声顺序
    """

    def __init__(self, noise_types=None, shuffle=False):
        self.noise_types = noise_types or []
        self.shuffle = shuffle

    def __call__(self, img):
        if self.shuffle:
            random.shuffle(self.noise_types)

        for noise in self.noise_types:
            img = noise(img)

        return img

    @classmethod
    def from_config(cls, config):
        """
        从配置创建噪声增强组合

        Args:
            config: 噪声配置字典

        Returns:
            NoiseCompose: 噪声增强组合
        """
        noise_list = []

        noise_cfg = config.get("noise", {})

        if noise_cfg.get("gaussian_prob", 0) > 0:
            noise_list.append(GaussianNoise(
                std=tuple(noise_cfg.get("gaussian_std", [0.01, 0.05])),
                p=noise_cfg["gaussian_prob"]
            ))

        if noise_cfg.get("salt_pepper_prob", 0) > 0:
            noise_list.append(SaltPepperNoise(
                amount=noise_cfg.get("salt_pepper_amount", 0.02),
                p=noise_cfg["salt_pepper_prob"]
            ))

        if noise_cfg.get("speckle_prob", 0) > 0:
            noise_list.append(SpeckleNoise(
                std=noise_cfg.get("speckle_std", 0.05),
                p=noise_cfg["speckle_prob"]
            ))

        if noise_cfg.get("poisson_prob", 0) > 0:
            noise_list.append(PoissonNoise(p=noise_cfg["poisson_prob"]))

        return cls(noise_list, shuffle=False)


def get_geometric_transforms(config=None):
    """
    获取几何变换增强

    Args:
        config: 增强配置字典

    Returns:
        list: 几何变换列表
    """
    transforms_list = []

    if config is None:
        return transforms_list

    geo_cfg = config.get("geometric", {})

    if geo_cfg.get("horizontal_flip_prob", 0) > 0:
        transforms_list.append(
            transforms.RandomHorizontalFlip(p=geo_cfg["horizontal_flip_prob"])
        )

    if geo_cfg.get("vertical_flip_prob", 0) > 0:
        transforms_list.append(
            transforms.RandomVerticalFlip(p=geo_cfg["vertical_flip_prob"])
        )

    if geo_cfg.get("rotate_prob", 0) > 0:
        degrees = tuple(geo_cfg.get("rotate_degrees", [-15, 15]))
        transforms_list.append(
            transforms.RandomRotation(degrees, p=geo_cfg.get("rotate_prob", 0.3))
        )

    if geo_cfg.get("zoom_prob", 0) > 0:
        scale = tuple(geo_cfg.get("zoom_range", [0.9, 1.1]))
        transforms_list.append(
            transforms.RandomResizedCrop(
                size=(256, 256),
                scale=scale,
                ratio=(1.0, 1.0),
                p=geo_cfg["zoom_prob"]
            )
        )

    return transforms_list


def get_photometric_transforms(config=None):
    """
    获取光度变换增强

    Args:
        config: 增强配置字典

    Returns:
        list: 光度变换列表
    """
    transforms_list = []

    if config is None:
        return transforms_list

    photo_cfg = config.get("photometric", {})

    brightness = photo_cfg.get("brightness_range", [0.8, 1.2])
    contrast = photo_cfg.get("contrast_range", [0.8, 1.2])
    brightness_prob = photo_cfg.get("brightness_prob", 0)
    contrast_prob = photo_cfg.get("contrast_prob", 0)

    if brightness_prob > 0 or contrast_prob > 0:
        transforms_list.append(
            transforms.ColorJitter(
                brightness=(brightness[0], brightness[1]) if brightness_prob > 0 else 0,
                contrast=(contrast[0], contrast[1]) if contrast_prob > 0 else 0,
            )
        )

    if photo_cfg.get("blur_prob", 0) > 0:
        transforms_list.append(RandomBlur(
            kernel_sizes=tuple(photo_cfg.get("blur_kernel", [3, 5])),
            p=photo_cfg["blur_prob"]
        ))

    return transforms_list


def build_augmentation_pipeline(config, mode="train", image_size=(256, 256)):
    """
    构建完整的数据增强流水线

    Args:
        config: 增强配置字典
        mode: "train" 或 "val"
        image_size: 图像尺寸 [height, width]

    Returns:
        transforms.Compose: 增强流水线
    """
    pipeline_list = []

    if mode == "train" and config.get("enable", False):
        # 几何变换（作用于 PIL Image）
        pipeline_list.extend(get_geometric_transforms(config))
        # 光度变换（作用于 PIL Image）
        pipeline_list.extend(get_photometric_transforms(config))

    # 调整大小
    pipeline_list.append(transforms.Resize(image_size))
    pipeline_list.append(transforms.ToTensor())

    # 噪声增强（作用于 Tensor）
    if mode == "train" and config.get("enable", False):
        noise_transform = NoiseCompose.from_config(config)
        if noise_transform.noise_types:
            pipeline_list.append(noise_transform)

    # 标准化
    pipeline_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return transforms.Compose(pipeline_list)
