"""
预测脚本。
用途: 使用 train.py 保存的模型，对新的 weatherAUS 格式 CSV 输出降雨预测结果。
示例:
    python predict.py --checkpoint outputs/mlp_model.pt --csv-path weatherAUS.csv
"""

import argparse                                          # 命令行参数解析
from pathlib import Path

import pandas as pd
import torch

from data import load_prediction_data, resolve_data_path  # 数据加载与路径解析
from evaluate import load_checkpoint, load_preprocessor, predict_probabilities, resolve_preprocessor_path  # 复用评估工具
from model import build_model                            # 模型构建


def predict_from_checkpoint(checkpoint_path, csv_path, output_path, threshold_override=None):
    """加载模型和预处理器，对 CSV 中的样本进行预测并保存结果。"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)        # 加载 checkpoint

    preprocessor_path = resolve_preprocessor_path(checkpoint, checkpoint_path)  # 定位预处理器
    preprocessor = load_preprocessor(preprocessor_path) # 加载预处理器

    # original_df 用于保留原始输入列，feature_df 只用于模型预测
    original_csv_path = resolve_data_path(csv_path)      # 解析数据路径
    original_df = pd.read_csv(original_csv_path)         # 读取原始数据（保留所有列）
    feature_df = load_prediction_data(csv_path)          # 整理成模型输入特征表

    X_processed = preprocessor.transform(feature_df)    # 用预处理器转换特征
    if X_processed.shape[1] != checkpoint["input_dim"]: # 校验特征维度是否匹配训练时
        raise ValueError(                                # 不匹配则报错
            "预测数据的特征维度与训练模型不一致:"
            f"当前 {X_processed.shape[1]},模型需要 {checkpoint['input_dim']}。"
        )

    X_t = torch.tensor(X_processed, dtype=torch.float32)  # 转张量

    model = build_model(                                 # 重建模型结构
        model_name=checkpoint["model_name"],
        input_dim=checkpoint["input_dim"],
        dropout_p=checkpoint["dropout_p"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])  # 载入参数

    # 未指定 threshold_override 则使用 checkpoint 中保存的阈值
    threshold = checkpoint["threshold"] if threshold_override is None else threshold_override
    probabilities = predict_probabilities(model, X_t)   # 预测概率
    predictions = (probabilities >= threshold).astype(int)  # 按阈值转为 0/1

    result_df = original_df.copy()                       # 在原始数据上追加预测结果
    result_df["RainTomorrow_probability"] = probabilities  # 概率列
    result_df["RainTomorrow_pred"] = predictions        # 预测标签列（0/1）
    result_df["RainTomorrow_pred_label"] = result_df["RainTomorrow_pred"].map({0: "No", 1: "Yes"})  # 文字标签

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")  # 保存为 CSV（带 BOM，Excel 友好）

    print(f"预测模型:{checkpoint['model_name']}")        # 打印模型名
    print(f"分类阈值:{threshold}")                       # 打印阈值
    print(f"预测样本数:{len(result_df)}")                # 打印样本数
    print(f"预测结果已保存到:{output_path}")
    print("\n预测结果预览:")
    print(result_df[["RainTomorrow_probability", "RainTomorrow_pred", "RainTomorrow_pred_label"]].head())  # 预览前几行

    return result_df                                    # 返回结果表


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用已训练模型预测 RainTomorrow。")
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py 保存的 .pt 模型文件路径。")
    parser.add_argument("--csv-path", type=str, default="weatherAUS.csv", help="需要预测的 CSV 文件路径。")
    parser.add_argument("--output", type=str, default="outputs/predictions.csv", help="预测结果保存路径。")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="可选:覆盖 checkpoint 中保存的分类阈值。默认使用训练时的阈值。",
    )
    return parser.parse_args()


def main():
    args = parse_args()                                  # 解析参数
    predict_from_checkpoint(                             # 执行预测
        checkpoint_path=Path(args.checkpoint),
        csv_path=args.csv_path,
        output_path=Path(args.output),
        threshold_override=args.threshold,
    )


if __name__ == "__main__":                              # 作为脚本运行时
    main()                                              # 调用主函数