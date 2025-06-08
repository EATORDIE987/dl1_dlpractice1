import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


# * ------------------------------------ 删除标签y为缺失值的数据 ------------------------------------------------
def Delete_missing_y(df_train: pd.DataFrame):
    """
    删除训练集缺失的y

    参数：

    df_train：训练集

    返回：

    df_train_delete_missing_y：删除缺失的y后的训练集

    delete_missing_y：删除的行的信息
    """
    # todo 复制一份防止改变原参数
    df = df_train.copy()
    # todo 只找出y不缺失的数据构成我们的数据集
    # ! y缺失的数据应当删除而不是填充！！！
    df_train_delete_missing_y = df[df.iloc[:, df.shape[1] - 1].notna()]
    delete_missing_y = df[df.iloc[:, df.shape[1] - 1].isna()]
    # todo 输出删去后的数据集和删去的数据
    return df_train_delete_missing_y, delete_missing_y


# * ------------------------------------ 计算每列缺失值个数和缺失值占比 ------------------------------------------------
def Count_missing_data(
    df: pd.DataFrame, Obj: str = "omission"  # 如果 Obj 可以为 None，则用 Optional[str]
) -> pd.DataFrame:
    """
    统计并返回 Pandas DataFrame 中每列的缺失值数量和占所有数据的比重。

    根据 `Obj` 参数的设置，此函数可以返回所有列的缺失值统计，
    或者只返回那些实际包含缺失值的列的统计。

    参数:

    df : pd.DataFrame：
        需要进行缺失值统计的目标 DataFrame。

    Obj : str, 可选, 默认值为 'omission'
        一个字符串参数，用于控制输出内容的模式：
        - 'omission': (默认值) 只返回那些包含至少一个缺失值的列及其对应的缺失值数量。
            不包含缺失值的列将不会出现在结果中。
        - 'total': 返回 DataFrame 中所有列及其对应的缺失值数量。
            如果某列没有缺失值，其对应的缺失值数量将为 0。

    返回: pd.Series,一个 Pandas Series 对象：
        - 索引 (Index): DataFrame 的列名。
        - 值 (Values): 对应列中缺失值的数量。

        根据 `Obj` 参数的设置，这个 Series 可能只包含有缺失值的列，或者包含所有列。

    异常: ValueError：
        如果 `Obj` 参数的值不是 'omission' 或 'total' 中的任意一个。
    """
    # todo 计算所有列的缺失值数量和占比
    if Obj == "total":
        NANCount_Totalcolumns = df.isna().sum(axis=0)
        Percent = NANCount_Totalcolumns / len(df)
        return pd.concat(
            [
                NANCount_Totalcolumns,
                pd.Series([len(df)] * len(Percent), index=Percent.index),
                Percent,
            ],
            axis=1,
            keys=["缺失值个数", "数据总条数", "缺失值占比"],
        )
    # todo 只计算有缺失值的列的缺失值数量和占比
    elif Obj == "omission":
        NANCount_Totalcolumns = df.isna().sum(axis=0)
        # todo 挑选出有缺失值的列，也就是缺失值个数不为0的列
        NANCount_For_columns_with_missingdata = NANCount_Totalcolumns[
            NANCount_Totalcolumns != 0
        ]
        # todo 计算缺失值占比
        # ? len(df)表示df的行数
        Percent = NANCount_For_columns_with_missingdata / len(df)
        # todo 返回缺失值信息
        return pd.concat(
            [
                NANCount_For_columns_with_missingdata,
                pd.Series([len(df)] * len(Percent), index=Percent.index),
                Percent,
            ],
            axis=1,
            keys=["缺失值个数", "数据总条数", "缺失值占比"],
        )
    else:
        raise ValueError("Obj must be 'omission' or 'total'. Please check your input.")


# * ------------------------------------ 删除缺失值占比高的特征 ------------------------------------------------
def Delete_HighPercent_MissingData(
    df_train: pd.DataFrame, df_test: pd.DataFrame, alpha: float = 0.5
):
    """
    在这一步前，请先删除训练集缺失的y！

    删除缺失值占比超过alpha的特征并输出删除列的信息

    参数：

    df_train：目标训练集表格

    df_test：测试集表格

    alpha：占比阈值

    返回：

    Delete_List：删除列的信息

    df_delete_train：删除后的训练集

    df_delete_test：删除后的测试集
    """
    # todo 复制一份数据防止原数据被改变
    df_delete_train = df_train.copy()
    df_delete_test = df_test.copy()
    # todo 计算缺失数据占总数据量的百分比
    count_missingdata_percentage = Count_missing_data(df_train, Obj="omission")
    # todo 找出占比大于alpha的特征的详细信息
    Delete_List = count_missingdata_percentage[
        count_missingdata_percentage["缺失值占比"] >= alpha
    ]
    # todo 删除这些缺失值占比过大的特征
    for column in Delete_List.index.tolist():
        df_delete_train.pop(column)
        df_delete_test.pop(column)

    # todo 返回删除后的新训练集和测试集，以及被删列的信息
    return Delete_List, df_delete_train, df_delete_test


# * ------------------------------------ 自动填充无意义的缺失值 ------------------------------------------------
def Fill_meaningless_misssingdata(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    method_numeric: str = "median",
    method_category: str = "mode",
    columns: List[int] = [],
):
    """
    填充没有实际意义的缺失值

    注意：要用训练集的统计量填充测试集，避免造成数据泄露！

    参数：

    df_train: 训练集

    df_test: 测试集

    method_numeric: 数值型数据填充方法，mean或median，默认median

    method_category: 分类型数据填充方法，默认只有mode

    columns: List[int] = [],要填充的列，默认为整个表格,注意是整数索引，str索引请使用转换函数
    应当是训练集和测试集的所有需要填充的列的并

    返回：

    df_train_copy：填充后的训练集

    df_test_copy：填充后的测试集
    """
    # todo 默认取表格的所有列的整数索引
    # ! 注意不包含y，否则会发生错误，因为维度不同
    if columns == []:
        columns = list(range(df_train.shape[1] - 1))
    # todo 复制一份参数以防原参数被改变
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    # todo 遍历选中的每一列
    for column in columns:
        # todo 若为数值型变量
        if pd.api.types.is_numeric_dtype(df_train_copy.iloc[:, column]):
            if method_numeric == "mean":
                # todo 用训练集平均数填充训练集
                mean_value = df_train_copy.iloc[:, column].mean()
                df_train_copy.iloc[:, column] = df_train_copy.iloc[:, column].fillna(
                    mean_value
                )
                # todo 用训练集平均数填充测试集
                # ! 不能用测试集统计量填充测试集，这相当于偷看了测试集数据的分布，属于数据泄露！！！
                df_test_copy.iloc[:, column] = df_test_copy.iloc[:, column].fillna(
                    mean_value
                )
            elif method_numeric == "median":
                # todo 同上，填充中位数
                median_value = df_train_copy.iloc[:, column].median()
                df_train_copy.iloc[:, column] = df_train_copy.iloc[:, column].fillna(
                    median_value
                )
                df_test_copy.iloc[:, column] = df_test_copy.iloc[:, column].fillna(
                    median_value
                )
            else:
                # todo 报错，字符串表示的填充类型错误
                raise ValueError("method_numeric must be 'mean' or 'median'")
        # todo 若为分类型变量
        elif pd.api.types.is_categorical_dtype(
            df_train_copy.iloc[:, column]
        ) or pd.api.types.is_object_dtype(df_train_copy.iloc[:, column]):
            # todo 使用众数填充
            if method_category == "mode":
                # ! 依旧同上，只能使用训练集众数填充测试集
                mode_value = df_train_copy.iloc[:, column].mode()[0]
                df_train_copy.iloc[:, column] = df_train_copy.iloc[:, column].fillna(
                    mode_value
                )
                df_test_copy.iloc[:, column] = df_test_copy.iloc[:, column].fillna(
                    mode_value
                )
            else:
                # todo 类型报错
                raise ValueError("method_category must be 'mode'")
        else:
            raise TypeError(f"Column {column} is neither numeric nor categorical.")
    return df_train_copy, df_test_copy


# * ------------------------------------ 数据标准化，归一化以及onehot编码 ------------------------------------------------
def OneHot_Encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_method: str = "standardize",
    keep_target_separate: bool = True,
):
    """
    自动对训练集和测试集进行onehot编码，并可能返回一个可复原y的函数。

    注意：默认现在已有的数值型数据不存在缺失值，分类型数据的缺失值均有实际意义！

    核心功能:
    1.  始终将特征(X)和目标(y)的分离作为第一步，确保预处理器只学习公共特征。
    2.  根据 `keep_target_separate` 参数决定y的处理方式:
        - `True` (默认): 为y创建一个独立的缩放器进行处理。返回分离的X和y。
        - `False`: 使用与X中数值特征相同的缩放器处理y，然后将其拼接回训练集。
    3.  自动识别X中的数值型和分类型特征并进行相应处理。
    4.  对分类型特征进行独热编码，并能处理测试集中的新类别以确保维度一致。
    5.  始终返回一个函数，用于将处理后的y值逆向转换为原始尺度。

    参数:
        train_df (pd.DataFrame): 原始训练数据集，最后一列为目标y。
        test_df (pd.DataFrame): 原始测试数据集，仅包含特征。
        numeric_method (str): 数值特征的处理方法。
            可选: 'standardize' (默认,标准化), 'normalize'(归一化), 'none'(不处理)。
        keep_target_separate (bool): 为true则不处理y，为false则按照和处理x一样的方式处理y，并返回一个复原y的函数。

    返回:
        - 如果 `keep_target_separate=True`:
            df_train：处理后的整个训练集，包含y
            X_test_processed：处理后的测试集
        - 如果 `keep_target_separate=False`:
            df_train：处理后的整个训练集，包含y
            X_test_processed：处理后的测试集
            inverse_transform_y：复原y的函数
    """
    # todo 复制一份参数，防止修改原数据
    train_df = train.copy()
    test_df = test.copy()
    # todo 将所有缺失值统一成字符串"missing_value"，方便后续处理（因为前面已经说过了，缺失值有意义）
    train_df = train_df.fillna(value=pd.NA).replace({pd.NA: "missing_value"})
    test_df = test_df.fillna(value=pd.NA).replace({pd.NA: "missing_value"})
    # todo 分离特征(X)和标签(y)
    target_name = train_df.columns[-1]
    y_train = train_df[target_name].copy()
    X_train = train_df.drop(columns=[target_name]).copy()
    X_test = test_df.copy()

    # todo 利用sklearn库创建数值型特征转换器，分为标准化，归一化，不变三种
    if numeric_method == "standardize":
        # ? 定义标准化特征转换器
        numeric_transformer = StandardScaler()
    elif numeric_method == "normalize":
        # ? 定义归一化特征转换器
        numeric_transformer = MinMaxScaler()
    elif numeric_method == "none":
        # ? passthrough的意思是“直接通过，不做任何处理”
        numeric_transformer = "passthrough"
    else:
        raise ValueError(
            "`numeric_method` 必须是 'standardize', 'normalize', 或 'none'"
        )

    # todo 创建分类型特征转换器（one-hot编码）
    categorical_transformer = Pipeline(
        steps=[
            # todo 以防万一，把空值填充缺失值
            (
                "imputer",
                SimpleImputer(
                    missing_values="", strategy="constant", fill_value="missing_value"
                ),
            ),
            # ! sklearn的OneHotEncoder函数和pandas的getdummies函数的区别在于：
            # !     当有特征在训练集里未出现，但在测试集里出现时，这时如果用pandas的函数分别处理训练集和测试集，会使处理后的训练集和测试集维度不统一
            # !     如果采用拼接训练集和测试集，对整个集合进行编码再分开的话，这实际上造成了数据泄露！！！
            # !         因为你'偷看'了测试集的数据的类型，把它放进了训练集里！！！
            # !     而sklearn的函数完美解决了这个问题，当它在测试集遇见在训练集没见过的函数时，不会创建一个新的编码列，而是把已有的关于这一项的编码列全赋值为0
            # !     这样，我们的训练就会相当诚实，没见过就是没见过，不会造成数据泄露！！！
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # todo 挑出训练集（除去了y）的数值型特征和分类型特征
    numeric_features: List[str] = X_train.select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_features: List[str] = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # todo 定义总特征转换器
    preprocessor = ColumnTransformer(
        # ? transformers参数是核心，它是一个列表，定义了要对哪些列做什么样的处理
        transformers=[
            # ? 第一个处理步骤，我们给它取名叫num,对数值型特征进行特征转换（标准化，归一化，不变）
            ("num", numeric_transformer, numeric_features),
            # ? 第二个处理步骤，我们给它取名叫cat，对分类型数据进行转换（onehot编码）
            ("cat", categorical_transformer, categorical_features),
        ],
        # ? remainder="passthrough": 这个参数用来处理那些既不在numeric_features也不在categorical_features列表中的其他所有列。
        # ? passthrough的意思是“直接通过，不做任何处理”。这些列将被原封不动地保留下来。
        # ? 另一个常见的选项是 drop，表示丢弃这些未指定的列。
        remainder="passthrough",
    )

    X_train_processed = pd.DataFrame(
        # ? 使用训练集（除去y）的数据拟合转换器并返回转换训练集的结果
        preprocessor.fit_transform(X_train),
        index=X_train.index,
        # ? preprocessor.get_feature_names_out()的作用是经过所有转换和处理之后，最终输出的特征的名称组成的列表。
        columns=preprocessor.get_feature_names_out(),
    )
    X_test_processed = pd.DataFrame(
        # ? 使用刚才训练集拟合的特征转换器转换测试集
        # ! 不能用测试集的特征拟合转换器，这是数据泄露！！！相当于偷看了测试集的数据分布！！！
        preprocessor.transform(X_test),
        index=X_test.index,
        columns=preprocessor.get_feature_names_out(),
    )

    # todo 处理y并创建复原函数
    y_scaler = None  # 初始化y的缩放器

    if keep_target_separate:
        # todo 若不处理y，则直接拼接去掉y的训练集和原来的y
        df_train = pd.concat([X_train_processed, y_train], axis=1)
        return df_train, X_test_processed
    else:
        # todo 若处理y，则定义一个新的与刚才处理x的数值型特征一样的特征转换器，专门用来转换y
        if numeric_method == "none":
            # todo 不处理，复原函数为本身，则mu=0，sigma=1
            y_scaler = "passthrough"
            y_train_processed = y_train.copy()
            mu = 0
            sigma = 1
        else:
            # todo 创建一个同类型的独立scaler
            y_scaler = numeric_transformer.__class__()
            # todo 用y拟合特征转换器
            y_train_processed_np = y_scaler.fit_transform(y_train.to_frame())
            if numeric_method == "standardize":
                # todo 标准化取方差和均值作为复原函数参数
                mu = y_scaler.mean_
                sigma = y_scaler.scale_
            else:
                # todo 归一化取最小值和最大值与最小值的差为复原函数参数
                mu = y_scaler.data_min_
                sigma = y_scaler.data_range_
            # todo y的最终处理值
            y_train_processed = pd.Series(
                y_train_processed_np.flatten(), index=y_train.index, name=y_train.name
            )
        # todo 将处理后的y拼接到处理后的训练集特征中，使用原始y的列名
        df_train = X_train_processed.copy()
        df_train[target_name] = y_train_processed

        # todo 定义复原函数，将处理后的特征变为原本特征
        def inverse_transform_y(y):
            return y * sigma + mu

        return df_train, X_test_processed, sigma,mu


# * ------------------------------------ 列名列表转换为整数索引列表 ------------------------------------------------
def nameindex_switchto_intindex(df: pd.DataFrame, columns_name_list: List) -> List:
    """
    将列名列表转化为整数索引列表

    参数：
    df：目标表格
    columns_name_list：列名列表

    返回：
    intindexList：整数索引列表
    """
    intindexList = []
    for name in columns_name_list:
        # ? df.columns.get_loc(name)表示的是列名为name的列在表格df中的整数列索引
        intindex = df.columns.get_loc(name)
        intindexList.append(intindex)
    return intindexList


# * ------------------------------------ 调试代码功能 ------------------------------------------------
if __name__ == "__main__":
    # todo 测试库功能
    df1 = pd.read_excel("train.xlsx", index_col="Id")
    df1.pop("Unnamed: 0")
    df2 = pd.read_excel("test.xlsx", index_col="Id")
    df2.pop("Unnamed: 0")
    # todo 有无意义缺失值的列
    columnname_meaningless_data = [
        "MSZoning",
        "LotFrontage",
        "Utilities",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "KitchenQual",
        "Functional",
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",
        "SaleType",
        "Electrical",
    ]
    info, df111, df222 = Delete_HighPercent_MissingData(df1, df2, 0.5)
    df11, df22 = Fill_meaningless_misssingdata(
        df111,
        df222,
        columns=nameindex_switchto_intindex(df111, columnname_meaningless_data),
    )
    df1111, df2222, y = OneHot_Encode(
        df11, df22, keep_target_separate=False, numeric_method="standardize"
    )
    """
    print(df1111, "\n", df2222)
    print(y(df1111.iloc[:, -1].values))"""

    # todo 测试有没有编码缺失值
    train_data = {
        "Age": [28, 35, 45, 23, 51],
        "Salary": [70000.0, 95000.0, 120000.0, 45000.0, 135000.0],
        "Department": ["Sales", "IT", np.nan, "Sales", "HR"],  # 分类型变量，包含NaN
        "City": [
            "Tokyo",
            "Osaka",
            "Kyoto",
            np.nan,
            "Tokyo",
        ],  # 另一个分类型变量，包含NaN
        "Satisfaction_Score": [4.5, 4.8, 4.0, 3.5, 4.9],  # 目标变量y
    }

    train_df = pd.DataFrame(train_data)

    print("----------- 训练集 -----------")
    print(train_df)
    print("----------- 测试集 -----------")

    # --- 创建测试集DataFrame (test_df) ---

    # 包含了训练集中没有的新类别 'Marketing'
    test_data = {
        "Age": [31, 29, 48],
        "Salary": [82000.0, 68000.0, 150000.0],
        # 'Marketing' 是测试集中独有的新类别
        # 'HR' 是训练集中独有的类别，这里没有
        "Department": ["IT", "Marketing", "Sales"],
        "City": ["Osaka", "Kyoto", "Tokyo"],
    }

    test_df = pd.DataFrame(test_data)
    print(test_df)
    new1, new2, y = OneHot_Encode(
        train_df, test_df, keep_target_separate=False, numeric_method="standardize"
    )
    new1.to_excel("1.xlsx")
    new2.to_excel("2.xlsx")
