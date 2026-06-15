
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双卷积块: (Conv2d -> BatchNorm -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样块: MaxPool -> DoubleConv
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样块: 上采样 -> 拼接 -> DoubleConv
    """

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 弱监督分割骨干网络

    支持图像级弱监督学习，仅使用图像级别标签（有/无缺陷）进行训练。
    通过分类头（Classification Head）提供图像级监督信号，
    分割头（Segmentation Head）输出像素级掩码。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（分割类别数）
        base_channels: 基础通道数
        bilinear: 是否使用双线性插值上采样
        dropout_rate: Dropout 比率
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64,
                 bilinear=True, dropout_rate=0.3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # 编码器（下采样路径）
        self.inc = DoubleConv(in_channels, base_channels, dropout_rate=dropout_rate)
        self.down1 = Down(base_channels, base_channels * 2, dropout_rate=dropout_rate)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout_rate=dropout_rate)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout_rate=dropout_rate)

        # 解码器（上采样路径）
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout_rate=dropout_rate)

        # 分割头 - 输出像素级掩码
        self.outc = OutConv(base_channels, out_channels)

        # 分类头 - 图像级分类（用于弱监督）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 16 // factor, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

        # 注意力门控（可选，用于弱监督定位）
        self.attention_gate = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            dict: 包含以下键的字典
                - mask: 分割掩码 logits [B, out_channels, H, W]
                - class_logit: 图像级分类 logit [B, 1]
                - attention: 注意力图 [B, 1, H, W]
        """
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 图像级分类特征
        class_logit = self.classifier(x5)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 分割输出
        mask_logit = self.outc(x)

        # 注意力图（用于弱监督定位解释）
        attention = self.attention_gate(x)

        return {
            "mask": mask_logit,
            "class_logit": class_logit,
            "attention": attention,
        }

    def get_seg_mask(self, x, threshold=0.5):
        """
        获取二值分割掩码

        Args:
            x: 输入图像
            threshold: 二值化阈值

        Returns:
            torch.Tensor: 二值掩码 [B, 1, H, W]
        """
        outputs = self.forward(x)
        mask = torch.sigmoid(outputs["mask"])
        return (mask > threshold).float()

    @classmethod
    def from_config(cls, config):
        """
        从配置字典创建模型

        Args:
            config: 模型配置字典

        Returns:
            UNet: 模型实例
        """
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 1),
            base_channels=config.get("base_channels", 64),
            bilinear=config.get("bilinear", True),
            dropout_rate=config.get("dropout_rate", 0.3),
        )


def build_model(config):
    """
    根据配置构建模型

    Args:
        config: 配置字典

    Returns:
        nn.Module: 模型实例
    """
    model_name = config.get("model", {}).get("name", "unet")

    if model_name.lower() == "unet":
        return UNet.from_config(config["model"])
    else:
        raise ValueError(f"未知的模型类型: {model_name}")
