"""
训练入口脚本。

示例：
    python train.py --model mlp
    python train.py --model cnn --epochs 80 --batch-size 512
    python train.py --model wide_mlp --output-dir outputs
"""
import argparse                                          # 命令行参数解析
import json                                              # 保存指标为 JSON
import pickle                                            # 序列化预处理器
from pathlib import Path

import numpy as np                                       # 数值计算
import torch
import torch.nn as nn
import torch.optim as optim                              # 优化器

from data import prepare_train_val_data                  # 数据准备函数
from evaluate import compute_binary_metrics, predict_probabilities, print_metrics  # 评估工具
from model import available_models, build_model          # 模型工具


RANDOM_STATE = 42                                        # 全局随机种子默认值


def set_random_seed(seed: int = RANDOM_STATE):
    """固定随机种子，便于复现实验。"""
    np.random.seed(seed)                                 # 固定 numpy 随机性
    torch.manual_seed(seed)                              # 固定 torch 随机性


def make_batches(X, y, batch_size: int):
    """按 batch_size 生成小批量训练数据。"""
    for start in range(0, len(X), batch_size):           # 按步长遍历
        end = start + batch_size                         # 计算批次结束位置
        yield X[start:end], y[start:end]                 # 产出一个批次


def train_model(model, X_train_t, y_train_t, X_val_t, y_val_t,
                pos_weight, learning_rate, weight_decay,
                num_epochs, batch_size, patience):
    """训练模型，并用验证集损失进行早停。"""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 带类别权重的二分类损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam 优化器

    best_val_loss = float("inf")                         # 记录最优验证损失，初始为无穷大
    best_state = None                                    # 记录最优模型参数
    no_improve_count = 0                                 # 连续未改善的轮数

    for epoch in range(1, num_epochs + 1):               # 逐轮训练
        model.train()                                    # 切换到训练模式
        train_loss = 0.0                                 # 累计训练损失

        # 每轮打乱训练样本，避免模型学到样本顺序
        permutation = torch.randperm(X_train_t.size(0))  # 随机排列索引
        X_train_shuffled = X_train_t[permutation]        # 打乱特征
        y_train_shuffled = y_train_t[permutation]        # 打乱标签

        for X_batch, y_batch in make_batches(X_train_shuffled, y_train_shuffled, batch_size):
            optimizer.zero_grad()                        # 清空梯度
            logits = model(X_batch)                      # 前向计算
            loss = criterion(logits, y_batch)            # 计算损失
            loss.backward()                              # 反向传播
            optimizer.step()                             # 更新参数
            train_loss += loss.item() * X_batch.size(0)  # 累加损失（按样本数加权）

        train_loss /= len(X_train_t)                     # 计算平均训练损失

        model.eval()                                     # 切换到评估模式
        with torch.no_grad():                            # 关闭梯度
            val_logits = model(X_val_t)                  # 验证集前向
            val_loss = criterion(val_logits, y_val_t).item()  # 验证损失

        if epoch == 1 or epoch % 10 == 0:                # 第 1 轮或每 10 轮打印一次
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:                     # 验证损失改善
            best_val_loss = val_loss                     # 更新最优损失
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  # 保存当前参数
            no_improve_count = 0                         # 重置计数
        else:                                            # 未改善
            no_improve_count += 1                        # 计数加 1
            if no_improve_count >= patience:             # 达到早停耐心
                print(f"验证集损失连续 {patience} 个 epoch 没有改善,提前停止。")
                break                                    # 结束训练

    if best_state is not None:                           # 若有最优参数
        model.load_state_dict(best_state)                # 恢复最优参数

    return model                                         # 返回训练好的模型


def save_artifacts(output_dir, model_name, model, preprocessor, metrics, args, input_dim):
    """保存模型、预处理器和评估指标。"""
    output_dir.mkdir(parents=True, exist_ok=True)        # 创建输出目录

    checkpoint_path = output_dir / f"{model_name}_model.pt"          # 模型文件路径
    preprocessor_path = output_dir / f"{model_name}_preprocessor.pkl"  # 预处理器路径
    metrics_path = output_dir / f"{model_name}_metrics.json"         # 指标文件路径

    with open(preprocessor_path, "wb") as f:             # 保存预处理器
        pickle.dump(preprocessor, f)

    checkpoint = {                                       # 组装 checkpoint 字典
        "model_name": model_name,                        # 模型名
        "model_state_dict": model.state_dict(),          # 模型参数
        "input_dim": input_dim,                          # 输入维度
        "dropout_p": args.dropout,                       # dropout 概率
        "threshold": args.threshold,                     # 分类阈值
        "random_state": args.random_state,               # 随机种子
        "test_size": args.test_size,                     # 验证集比例
        "preprocessor_path": preprocessor_path.name,     # 预处理器文件名
    }
    torch.save(checkpoint, checkpoint_path)              # 保存 checkpoint

    with open(metrics_path, "w", encoding="utf-8") as f: # 保存指标
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存到:{checkpoint_path}")
    print(f"预处理器已保存到:{preprocessor_path}")
    print(f"评估指标已保存到:{metrics_path}")

    return checkpoint_path                               # 返回模型路径


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 weatherAUS 降雨二分类模型。")
    parser.add_argument("--csv-path", type=str, default="weatherAUS.csv", help="数据文件路径。")
    parser.add_argument("--model", type=str, default="mlp", choices=available_models(), help="选择模型。")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数。")
    parser.add_argument("--batch-size", type=int, default=256, help="小批量大小。")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心轮数。")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正则强度。")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout 概率。")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值。")
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集比例。")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="随机种子。")
    parser.add_argument("--output-dir", type=str, default="outputs", help="保存目录。")
    return parser.parse_args()


def main():
    args = parse_args()                                  # 解析参数
    set_random_seed(args.random_state)                   # 固定随机种子

    data = prepare_train_val_data(                       # 准备数据
        csv_path=args.csv_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    df = data["df"]                                      # 完整数据
    X_train = data["X_train"]                            # 训练特征
    X_val = data["X_val"]                                # 验证特征
    y_train = data["y_train"]                            # 训练标签
    y_val = data["y_val"]                                # 验证标签

    print(f"数据规模:{df.shape[0]} 行,{df.shape[1]} 列")  # 打印数据规模
    print("目标变量分布:")
    print(df["RainTomorrow"].value_counts().rename(index={0: "No", 1: "Yes"}))  # 打印标签分布
    print(f"\n当前训练模型:{args.model}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)             # 转张量
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # 标签转列向量
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # 正类 Yes 较少，pos_weight 降低模型只预测 No 的偏向
    negative_count = (y_train == 0).sum()                # 负类样本数
    positive_count = (y_train == 1).sum()                # 正类样本数
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)  # 正类权重

    model = build_model(                                 # 构建模型
        model_name=args.model,
        input_dim=X_train_t.shape[1],
        dropout_p=args.dropout,
    )
    model = train_model(                                 # 训练模型
        model=model,
        X_train_t=X_train_t,                             # 训练特征张量
        y_train_t=y_train_t,                             # 训练标签张量
        X_val_t=X_val_t,                                 # 验证特征张量
        y_val_t=y_val_t,                                 # 验证标签张量
        pos_weight=pos_weight,                           # 正类权重
        learning_rate=args.lr,                           # 学习率
        weight_decay=args.weight_decay,                  # L2 正则
        num_epochs=args.epochs,                          # 最大轮数
        batch_size=args.batch_size,                      # 批大小
        patience=args.patience,                          # 早停耐心
    )

    probabilities = predict_probabilities(model, X_val_t)  # 在验证集上预测概率
    metrics = compute_binary_metrics(y_val, probabilities, threshold=args.threshold)  # 计算评估指标
    print_metrics(metrics)                               # 打印指标

    save_artifacts(                                      # 保存模型、预处理器、指标
        output_dir=Path(args.output_dir),               # 输出目录
        model_name=args.model,                           # 模型名
        model=model,                                     # 训练好的模型
        preprocessor=data["preprocessor"],              # 预处理器
        metrics=metrics,                                 # 评估指标
        args=args,                                       # 命令行参数
        input_dim=X_train_t.shape[1],                    # 输入维度
    )


if __name__ == "__main__":                              # 作为脚本运行时
    main()                                              # 调用主函数