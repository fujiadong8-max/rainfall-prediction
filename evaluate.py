"""
模型评估脚本。
训练后可单独运行: python evaluate.py --checkpoint outputs/mlp_model.pt
"""

import argparse                                          # 命令行参数解析
import json                                              # 读写 JSON
import pickle                                            # 反序列化预处理器
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score  # 各类评估指标

from data import prepare_train_val_data                  # 数据准备函数
from model import build_model                            # 模型构建函数


def compute_binary_metrics(y_true, probabilities, threshold=0.5):
    """根据预测概率计算二分类指标。"""
    predictions = (probabilities >= threshold).astype(int)  # 概率超过阈值判为 1

    return {                                             # 返回指标字典
        "roc_auc": float(roc_auc_score(y_true, probabilities)),  # ROC AUC（用概率算）
        "accuracy": float(accuracy_score(y_true, predictions)),  # 准确率（用预测标签算）
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),  # 混淆矩阵
        "classification_report": classification_report(  # 分类报告（精确率/召回率/F1）
            y_true,
            predictions,
            target_names=["No", "Yes"],                  # 标签名称
            output_dict=True,                            # 以字典形式返回
            zero_division=0,                             # 分母为 0 时返回 0，避免报错
        ),
    }


def print_metrics(metrics):
    """以便于阅读的格式打印评估结果。"""
    print("\n验证集评估结果")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")         # 打印 AUC
    print(f"Accuracy: {metrics['accuracy']:.4f}")        # 打印准确率
    print("\n混淆矩阵:")
    print(np.array(metrics["confusion_matrix"]))         # 打印混淆矩阵

    report = metrics["classification_report"]            # 取出分类报告
    print("\n分类报告:")
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")  # 表头
    for label in ["No", "Yes", "macro avg", "weighted avg"]:  # 逐行打印每个类别和平均
        row = report[label]
        print(
            f"{label:>10} "
            f"{row['precision']:>10.2f} "                # 精确率
            f"{row['recall']:>10.2f} "                   # 召回率
            f"{row['f1-score']:>10.2f} "                 # F1 分数
            f"{row['support']:>10.0f}"                   # 样本数
        )


def predict_probabilities(model, X_t):
    """输出 RainTomorrow=Yes 的预测概率。"""
    model.eval()                                         # 切换评估模式（关闭 Dropout 等）
    with torch.no_grad():                                # 关闭梯度计算
        logits = model(X_t)                              # 前向得到 logit
        probabilities = torch.sigmoid(logits).cpu().numpy().ravel()  # sigmoid 转概率并展平为一维
    return probabilities


def load_checkpoint(checkpoint_path):
    """加载训练脚本保存的 checkpoint。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():                     # 文件不存在
        raise FileNotFoundError(f"找不到模型文件:{checkpoint_path}")  # 抛异常
    return torch.load(checkpoint_path, map_location="cpu")  # 加载到 CPU


def load_preprocessor(preprocessor_path):
    """加载训练阶段保存的预处理器。"""
    with open(preprocessor_path, "rb") as f:             # 二进制读取
        return pickle.load(f)                            # 反序列化返回


def resolve_preprocessor_path(checkpoint, checkpoint_path):
    """根据 checkpoint 中的记录找到预处理器文件。"""
    preprocessor_path = Path(checkpoint["preprocessor_path"])  # 取出记录的路径
    if not preprocessor_path.is_absolute():              # 若是相对路径，尝试多个候选位置
        cwd_candidate = preprocessor_path                # 候选 1：当前工作目录
        same_dir_candidate = checkpoint_path.parent / preprocessor_path.name  # 候选 2：与模型同目录
        nested_candidate = checkpoint_path.parent / preprocessor_path         # 候选 3：模型目录下的嵌套路径

        if cwd_candidate.exists():                       # 候选 1 存在
            preprocessor_path = cwd_candidate
        elif same_dir_candidate.exists():                # 候选 2 存在
            preprocessor_path = same_dir_candidate
        else:                                            # 都不存在则用候选 3
            preprocessor_path = nested_candidate

    return preprocessor_path                             # 返回最终路径


def evaluate_from_checkpoint(checkpoint_path, csv_path="weatherAUS.csv"):
    """从 checkpoint 加载模型，并在同一验证集划分上重新评估。"""
    checkpoint = load_checkpoint(checkpoint_path)        # 加载 checkpoint
    preprocessor_path = resolve_preprocessor_path(checkpoint, checkpoint_path)  # 定位预处理器
    preprocessor = load_preprocessor(preprocessor_path) # 加载预处理器

    # 使用训练时相同的 test_size 和 random_state，保证验证集划分一致
    data = prepare_train_val_data(
        csv_path=csv_path,
        test_size=checkpoint["test_size"],               # 复用训练时的验证集比例
        random_state=checkpoint["random_state"],         # 复用训练时的随机种子
        preprocessor=preprocessor,                       # 复用已保存的预处理器
        fit_preprocessor=False,                          # 评估阶段不重新拟合
    )

    X_val_t = torch.tensor(data["X_val"], dtype=torch.float32)  # 验证特征转张量
    y_val = data["y_val"]                                # 验证标签

    model = build_model(                                 # 按 checkpoint 信息重建模型
        model_name=checkpoint["model_name"],
        input_dim=checkpoint["input_dim"],
        dropout_p=checkpoint["dropout_p"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])  # 载入训练好的参数

    probabilities = predict_probabilities(model, X_val_t)  # 预测概率
    metrics = compute_binary_metrics(y_val, probabilities, threshold=checkpoint["threshold"])  # 计算指标
    print_metrics(metrics)                               # 打印指标

    return metrics                                       # 返回指标


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="加载已训练模型并评估 weatherAUS 验证集表现。")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练脚本保存的 .pt 模型文件路径。")
    parser.add_argument("--csv-path", type=str, default="weatherAUS.csv", help="weatherAUS.csv 文件路径。")
    parser.add_argument("--save-metrics", type=str, default=None, help="可选:保存评估指标 JSON 的路径。")
    return parser.parse_args()


def main():
    args = parse_args()                                  # 解析参数
    metrics = evaluate_from_checkpoint(Path(args.checkpoint), csv_path=args.csv_path)  # 评估

    if args.save_metrics:                                # 若指定了保存路径
        save_path = Path(args.save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
        with open(save_path, "w", encoding="utf-8") as f:    # 写入 JSON
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n评估指标已保存到:{save_path}")


if __name__ == "__main__":                              # 作为脚本运行时
    main()                                              # 调用主函数