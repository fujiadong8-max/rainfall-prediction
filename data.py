"""
weatherAUS 数据处理模块。

本模块只负责数据相关工作：
1. 读取 weatherAUS.csv
2. 清理目标变量 RainTomorrow
3. 划分训练集和验证集
4. 对数值变量和类别变量进行预处理
"""
from pathlib import Path                                  # 导入 Path，用于跨平台路径处理
from typing import Optional, Union                        # 导入类型注解工具

import pandas as pd                                       # 导入 pandas，用于读取和操作表格数据
from sklearn.compose import ColumnTransformer             # 列变换器：对不同列应用不同预处理
from sklearn.impute import SimpleImputer                  # 缺失值填补器
from sklearn.model_selection import train_test_split      # 划分训练/验证集
from sklearn.pipeline import Pipeline                     # 把多个处理步骤串成流水线
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 独热编码 + 标准化


def resolve_data_path(csv_path: Union[str, Path]) -> Path:
    """把相对路径解析到当前项目目录下。"""
    csv_path = Path(csv_path)                             # 把输入转成 Path 对象
    if csv_path.is_absolute():                            # 如果已经是绝对路径
        return csv_path                                  # 直接返回
    return Path(__file__).resolve().parent / csv_path    # 否则拼接到本文件所在目录下


def load_weather_data(csv_path="weatherAUS.csv") -> pd.DataFrame:
    """读取 weatherAUS.csv，并完成基础清理。"""
    csv_path = resolve_data_path(csv_path)               # 解析为绝对路径
    if not csv_path.exists():                            # 文件不存在
        raise FileNotFoundError(f"找不到数据文件:{csv_path}")  # 抛出异常

    df = pd.read_csv(csv_path)                            # 读取 CSV 为 DataFrame

    # RainTomorrow 是监督学习目标；目标缺失的样本无法用于训练或验证
    df = df.dropna(subset=["RainTomorrow"]).copy()       # 删除目标列为空的行,并复制保存

    # 把日期转成月份，保留季节性信息；完整日期不适合直接做数值特征
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # 转为日期类型，无法解析的置为 NaT
    df["Month"] = df["Date"].dt.month                    # 提取月份作为新特征
    df = df.drop(columns=["Date"])                       # 删除原始日期列

    # 二分类标签：明天下雨 Yes=1，不下雨 No=0
    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0}).astype(int)  # 映射为 0/1 整数

    return df                                            # 返回清理后的 DataFrame


def split_features_target(df: pd.DataFrame):
    """拆分特征矩阵 X 和目标变量 y。"""
    X = df.drop(columns=["RainTomorrow"])                # 特征：除目标外的所有列
    y = df["RainTomorrow"].to_numpy()                    # 目标：转为 numpy 数组
    return X, y                                          # 返回特征和标签


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """把原始表格整理成模型可接收的特征表（用于预测阶段）。"""
    df = df.copy()                                       # 复制，避免修改原数据

    # 预测阶段数据可能带标签也可能不带；标签列不应作为特征输入
    if "RainTomorrow" in df.columns:                     # 如果存在标签列
        df = df.drop(columns=["RainTomorrow"])           # 删除它

    # 与训练阶段保持一致：Date 只保留月份特征
    if "Date" in df.columns:                             # 如果存在日期列
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # 转日期
        df["Month"] = df["Date"].dt.month               # 提取月份
        df = df.drop(columns=["Date"])                   # 删除原日期列

    return df                                            # 返回特征表


def load_prediction_data(csv_path) -> pd.DataFrame:
    """读取待预测新数据，不要求存在 RainTomorrow 标签。"""
    csv_path = resolve_data_path(csv_path)               # 解析路径
    if not csv_path.exists():                            # 文件不存在
        raise FileNotFoundError(f"找不到数据文件:{csv_path}")  # 抛异常

    df = pd.read_csv(csv_path)                            # 读取 CSV
    return prepare_feature_frame(df)                     # 整理成特征表后返回


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """构建预处理器：数值列填补+标准化，类别列填补+独热编码。"""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()  # 数值列名列表
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()        # 类别列名列表

    numeric_transformer = Pipeline(steps=[               # 数值列处理流水线
        ("imputer", SimpleImputer(strategy="median")),   # 用中位数填补缺失
        ("scaler", StandardScaler()),                    # 标准化（零均值单位方差）
    ])

    categorical_transformer = Pipeline(steps=[           # 类别列处理流水线
        ("imputer", SimpleImputer(strategy="most_frequent")),  # 用众数填补缺失
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # 独热编码，忽略未知类别
    ])

    return ColumnTransformer(transformers=[              # 组合成列变换器
        ("num", numeric_transformer, numeric_features),  # 数值列应用数值流水线
        ("cat", categorical_transformer, categorical_features),  # 类别列应用类别流水线
    ])


def prepare_train_val_data(
    csv_path="weatherAUS.csv",
    test_size=0.2,
    random_state=42,
    preprocessor=None,
    fit_preprocessor=True,
):
    """读取数据，划分训练/验证集，并完成预处理。"""
    df = load_weather_data(csv_path)                     # 读取并清理数据
    X, y = split_features_target(df)                     # 拆分特征和标签

    # stratify=y 保证训练集和验证集中正负样本比例一致
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,                             # 验证集比例
        random_state=random_state,                       # 随机种子，保证可复现
        stratify=y,                                      # 按标签分层抽样
    )

    if preprocessor is None:                             # 未传入预处理器
        preprocessor = build_preprocessor(X_train)       # 新建一个

    # 训练阶段只能在训练集上 fit；验证集参与 fit 会造成数据泄漏
    if fit_preprocessor:                                 # 需要拟合（训练阶段）
        X_train_processed = preprocessor.fit_transform(X_train)  # 拟合并转换训练集
    else:                                                # 不拟合（评估阶段复用）
        X_train_processed = preprocessor.transform(X_train)      # 只转换

    X_val_processed = preprocessor.transform(X_val)      # 验证集只做转换

    return {                                             # 返回所有结果
        "df": df,                                        # 清理后的完整数据
        "X_train": X_train_processed,                    # 处理后的训练特征
        "X_val": X_val_processed,                        # 处理后的验证特征
        "y_train": y_train,                              # 训练标签
        "y_val": y_val,                                  # 验证标签
        "preprocessor": preprocessor,                    # 预处理器（供后续复用）
    }