
"""
PCB 弱监督缺陷分割项目

用于 PCB 光学质检产线的显微图像缺陷检测，
采用图像级弱监督学习实现精准的缺陷分割定位。

模块结构：
- models: U-Net 分割骨干网络
- losses: 弱监督损失函数
- augmentations: 图像噪声增强
- data: 数据集与数据加载
- inference: 批量推理
- evaluation: 评估指标
- api: FastAPI 推理接口
- utils: 工具函数（设备管理、可视化等）
- scripts: 训练、评估、推理脚本
- configs: 配置文件
"""

__version__ = "1.0.0"
