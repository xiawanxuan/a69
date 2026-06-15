
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageLevelLoss(nn.Module):
    """
    图像级分类损失

    使用整张图像的标签（有/无缺陷）进行监督，
    通过分类头的输出计算二值交叉熵损失。
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, class_logit, image_label):
        """
        Args:
            class_logit: 分类头输出 [B, 1]
            image_label: 图像级标签 [B] 或 [B, 1]，0=正常，1=有缺陷

        Returns:
            torch.Tensor: 损失标量
        """
        image_label = image_label.float().view(-1, 1)
        return self.bce(class_logit, image_label)


class MaskConstraintLoss(nn.Module):
    """
    掩码约束损失

    弱监督掩码损失，包含两部分：
    1. 正样本图像必须有一定比例的缺陷区域（避免全零预测）
    2. 负样本图像的缺陷区域应该尽可能小
    """

    def __init__(self, threshold=0.5, min_region_size=10):
        super().__init__()
        self.threshold = threshold
        self.min_region_size = min_region_size

    def forward(self, mask_logit, image_label):
        """
        Args:
            mask_logit: 掩码 logits [B, 1, H, W]
            image_label: 图像级标签 [B]

        Returns:
            torch.Tensor: 损失标量
        """
        mask_prob = torch.sigmoid(mask_logit)
        image_label = image_label.float().view(-1, 1, 1, 1)

        batch_size = mask_prob.size(0)
        total_pixels = mask_prob.size(2) * mask_prob.size(3)

        # 正样本：期望有一定比例的缺陷像素
        positive_mask = image_label
        positive_pred = mask_prob * positive_mask
        positive_ratio = positive_pred.sum() / (positive_mask.sum() + 1e-8) / total_pixels

        # 负样本：期望几乎没有缺陷像素
        negative_mask = 1 - image_label
        negative_pred = mask_prob * negative_mask
        negative_ratio = negative_pred.sum() / (negative_mask.sum() + 1e-8) / total_pixels

        # 正样本损失：惩罚缺陷区域太小（目标至少有 5% 的缺陷区域）
        target_ratio = 0.05
        positive_loss = torch.relu(target_ratio - positive_ratio) ** 2

        # 负样本损失：惩罚有缺陷区域
        negative_loss = negative_ratio ** 2

        return positive_loss + negative_loss


class ConsistencyLoss(nn.Module):
    """
    一致性损失

    对同一图像的不同增强版本，其预测结果应该保持一致。
    用于半监督/弱监督学习中的正则化。
    """

    def __init__(self):
        super().__init__()

    def forward(self, mask_logit_1, mask_logit_2):
        """
        Args:
            mask_logit_1: 第一种增强的掩码 logits [B, 1, H, W]
            mask_logit_2: 第二种增强的掩码 logits [B, 1, H, W]

        Returns:
            torch.Tensor: 一致性损失
        """
        prob_1 = torch.sigmoid(mask_logit_1)
        prob_2 = torch.sigmoid(mask_logit_2)
        return F.mse_loss(prob_1, prob_2)


class PseudoLabelLoss(nn.Module):
    """
    伪标签损失

    使用高置信度的预测作为伪标签，指导模型训练。
    对正样本图像，选取高置信度的缺陷区域作为正伪标签；
    对负样本图像，全部区域作为负伪标签。
    """

    def __init__(self, threshold=0.5, min_confidence=0.9):
        super().__init__()
        self.threshold = threshold
        self.min_confidence = min_confidence

    def forward(self, mask_logit, image_label):
        """
        Args:
            mask_logit: 掩码 logits [B, 1, H, W]
            image_label: 图像级标签 [B]

        Returns:
            torch.Tensor: 伪标签损失
        """
        mask_prob = torch.sigmoid(mask_logit)
        image_label = image_label.float().view(-1, 1, 1, 1)

        # 生成伪标签
        pseudo_label = torch.zeros_like(mask_prob)

        # 正样本图像：高置信度缺陷区域作为正伪标签
        positive_mask = image_label
        confident_positive = (mask_prob > self.min_confidence).float()
        pseudo_label += confident_positive * positive_mask

        # 负样本图像：全部作为负伪标签（置信度高的区域）
        negative_mask = 1 - image_label
        confident_negative = (mask_prob < (1 - self.min_confidence)).float()
        pseudo_label_neg = confident_negative * negative_mask * 0.0

        # 只计算高置信度像素的损失
        confident_mask = confident_positive * positive_mask + confident_negative * negative_mask

        if confident_mask.sum() == 0:
            return torch.tensor(0.0, device=mask_logit.device, dtype=mask_logit.dtype)

        loss = F.binary_cross_entropy_with_logits(
            mask_logit, pseudo_label, reduction="none"
        )
        loss = (loss * confident_mask).sum() / (confident_mask.sum() + 1e-8)

        return loss


class AttentionLoss(nn.Module):
    """
    注意力正则化损失

    鼓励注意力图与分类结果一致：
    - 有缺陷的图像，注意力图应该有较高的响应
    - 正常图像，注意力图应该较低
    """

    def __init__(self):
        super().__init__()

    def forward(self, attention, image_label):
        """
        Args:
            attention: 注意力图 [B, 1, H, W]
            image_label: 图像级标签 [B]

        Returns:
            torch.Tensor: 注意力损失
        """
        image_label = image_label.float().view(-1, 1, 1, 1)

        # 全局平均池化得到图像级注意力分数
        attn_score = F.adaptive_avg_pool2d(attention, 1).view(-1, 1)
        label_score = image_label.view(-1, 1)

        # 注意力分数应该与标签一致
        loss = F.binary_cross_entropy_with_logits(
            attn_score, label_score
        )

        return loss


class WeakSupLoss(nn.Module):
    """
    弱监督综合损失

    组合多种弱监督损失：
    - 图像级分类损失（主要监督信号）
    - 掩码约束损失（保证正样本有缺陷区域）
    - 伪标签损失（自训练）
    - 注意力正则化损失

    Args:
        image_loss_weight: 图像级损失权重
        mask_loss_weight: 掩码约束损失权重
        consistency_weight: 一致性损失权重
        threshold: 二值化阈值
        min_region_size: 最小区域大小
    """

    def __init__(self, image_loss_weight=1.0, mask_loss_weight=0.5,
                 consistency_weight=0.1, threshold=0.5, min_region_size=10):
        super().__init__()

        self.image_loss_weight = image_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.consistency_weight = consistency_weight
        self.threshold = threshold
        self.min_region_size = min_region_size

        self.image_level_loss = ImageLevelLoss()
        self.mask_constraint_loss = MaskConstraintLoss(threshold, min_region_size)
        self.pseudo_label_loss = PseudoLabelLoss(threshold)
        self.attention_loss = AttentionLoss()
        self.consistency_loss = ConsistencyLoss()

    def forward(self, outputs, image_label, with_pixel_mask=False, pixel_mask=None):
        """
        计算综合弱监督损失

        Args:
            outputs: 模型输出字典，包含 mask, class_logit, attention
            image_label: 图像级标签 [B]
            with_pixel_mask: 是否有像素级掩码（完全监督模式）
            pixel_mask: 像素级掩码 [B, 1, H, W]

        Returns:
            dict: 包含总损失和各分项损失的字典
        """
        mask_logit = outputs["mask"]
        class_logit = outputs["class_logit"]
        attention = outputs.get("attention", None)

        loss_dict = {}

        # 1. 图像级分类损失
        loss_img = self.image_level_loss(class_logit, image_label)
        loss_dict["image_loss"] = loss_img

        # 2. 如果有像素级标签，使用完全监督损失
        if with_pixel_mask and pixel_mask is not None:
            loss_seg = F.binary_cross_entropy_with_logits(mask_logit, pixel_mask)
            loss_dict["seg_loss"] = loss_seg
            total_loss = self.image_loss_weight * loss_img + loss_seg
        else:
            # 3. 掩码约束损失（弱监督）
            loss_mask = self.mask_constraint_loss(mask_logit, image_label)
            loss_dict["mask_loss"] = loss_mask

            # 4. 伪标签损失
            loss_pseudo = self.pseudo_label_loss(mask_logit, image_label)
            loss_dict["pseudo_loss"] = loss_pseudo

            total_loss = (
                self.image_loss_weight * loss_img
                + self.mask_loss_weight * loss_mask
                + self.consistency_weight * loss_pseudo
            )

        # 5. 注意力正则化损失（如果有注意力图）
        if attention is not None:
            loss_attn = self.attention_loss(attention, image_label)
            loss_dict["attention_loss"] = loss_attn
            total_loss = total_loss + 0.1 * loss_attn

        loss_dict["total_loss"] = total_loss

        return loss_dict

    @classmethod
    def from_config(cls, config):
        """
        从配置创建弱监督损失

        Args:
            config: 损失配置字典

        Returns:
            WeakSupLoss: 损失实例
        """
        return cls(
            image_loss_weight=config.get("image_loss_weight", 1.0),
            mask_loss_weight=config.get("mask_loss_weight", 0.5),
            consistency_weight=config.get("consistency_weight", 0.1),
            threshold=config.get("threshold", 0.5),
            min_region_size=config.get("min_region_size", 10),
        )


def build_loss(config):
    """
    根据配置构建损失函数

    Args:
        config: 配置字典

    Returns:
        nn.Module: 损失函数
    """
    loss_name = config.get("loss", {}).get("name", "weak_ce")

    if loss_name.lower() == "weak_ce":
        return WeakSupLoss.from_config(config["loss"])
    else:
        raise ValueError(f"未知的损失类型: {loss_name}")
