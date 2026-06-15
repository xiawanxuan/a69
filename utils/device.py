
import torch
import yaml
import os
from pathlib import Path


def get_device(config=None):
    """
    自动检测并返回最优计算设备

    Args:
        config: 配置字典，包含 device 配置

    Returns:
        torch.device: 计算设备
    """
    if config is not None and isinstance(config, dict):
        device_cfg = config.get("device", {})
        if not device_cfg.get("auto_detect", True):
            if device_cfg.get("use_gpu", True) and torch.cuda.is_available():
                gpu_id = device_cfg.get("gpu_id", 0)
                return torch.device(f"cuda:{gpu_id}")
            else:
                return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info(device=None):
    """
    获取设备详细信息

    Args:
        device: torch.device 对象

    Returns:
        dict: 设备信息字典
    """
    if device is None:
        device = get_device()

    info = {
        "device": str(device),
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        info.update({
            "cuda_device_name": torch.cuda.get_device_name(device.index if device.index else 0),
            "cuda_memory_total": torch.cuda.get_device_properties(device.index if device.index else 0).total_memory / 1024**3,
            "cuda_memory_allocated": torch.cuda.memory_allocated(device.index if device.index else 0) / 1024**3,
        })

    return info


def load_config(config_path):
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        dict: 配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config, save_path):
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed):
    """
    设置随机种子，保证实验可复现

    Args:
        seed: 随机种子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
