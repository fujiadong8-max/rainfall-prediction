"""
模型定义与模型注册表。
包含 logistic / mlp / wide_mlp / cnn 四种模型。
"""

from typing import Callable, Dict, List                  # 类型注解工具

import torch                                             # PyTorch 主库
import torch.nn as nn                                    # 神经网络模块


class LogisticNet(nn.Module):
    """单层线性二分类模型，相当于 PyTorch 版逻辑回归。"""

    def __init__(self, input_dim: int, dropout_p: float = 0.0):
        super().__init__()                               # 初始化父类
        self.net = nn.Linear(input_dim, 1)               # 一个线性层，输出 1 个 logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                               # 前向传播


class RainfallNet(nn.Module):
    """默认 MLP 二分类模型。"""

    def __init__(self, input_dim: int, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(                        # 顺序堆叠各层
            nn.Linear(input_dim, 128),                   # 输入维 -> 128
            nn.ReLU(),                                   # 激活函数
            nn.Dropout(dropout_p),                       # 随机失活，防过拟合
            nn.Linear(128, 64),                          # 全连接：128 -> 64
            nn.ReLU(),                                   # 激活函数
            nn.Dropout(dropout_p),                       # Dropout
            nn.Linear(64, 1),                            # 输出层：64 -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                               # 前向传播


class WideMLPNet(nn.Module):
    """更宽的 MLP，用于测试更大容量是否带来收益。"""

    def __init__(self, input_dim: int, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),                   # 输入 -> 256
            nn.BatchNorm1d(256),                         # 批归一化，稳定训练
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),                         # 256 -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),                          # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 1),                            # 64 -> 1 输出
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {  # 模型名称到类的映射表
    "logistic": LogisticNet,
    "mlp": RainfallNet,
    "wide_mlp": WideMLPNet,

}


def available_models() -> List[str]:
    """返回当前可选择的模型名称（排序后）。"""
    return sorted(MODEL_REGISTRY.keys())


def build_model(model_name: str, input_dim: int, dropout_p: float = 0.2) -> nn.Module:
    """根据名称创建模型实例。"""
    if model_name not in MODEL_REGISTRY:                 # 名称不在注册表中
        names = ", ".join(available_models())            # 列出可选模型
        raise ValueError(f"未知模型:{model_name}。可选模型:{names}")  # 抛异常

    model_cls = MODEL_REGISTRY[model_name]               # 取出对应模型类
    return model_cls(input_dim=input_dim, dropout_p=dropout_p)  # 实例化并返回