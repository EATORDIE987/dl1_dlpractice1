import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any


def Count_missing_data(
    df: pd.DataFrame, Obj: str = "omission"  # 如果 Obj 可以为 None，则用 Optional[str]
) -> pd.Series:
    """
    统计并返回 Pandas DataFrame 中每列的缺失值数量。

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
    if Obj == "total":
        NANCount_Totalcolumns = df.isna().sum(axis=0)
        return NANCount_Totalcolumns
    elif Obj == "omission":
        NANCount_Totalcolumns = df.isna().sum(axis=0)
        NANCount_For_columns_with_missingdata = NANCount_Totalcolumns[
            NANCount_Totalcolumns != 0
        ]
        return NANCount_For_columns_with_missingdata
    else:
        raise ValueError("错误：Obj 参数值无效！应为 'omission' 或 'total'!")


def Fill_meaningless_missing_data(
    df1: pd.DataFrame,  # 第一个 DataFrame，必需参数
    df2: Optional[pd.DataFrame] = None,  # 第二个 DataFrame，可选参数，默认为 None
    method_for_numeric: Optional[
        str
    ] = "mean",  # 数值型数据处理方法，字符串，默认为 'mean'
    method_for_category: Optional[
        str
    ] = "mode",  # 类别型数据处理方法，字符串，默认为 'mode'
    columns: Optional[
        List[int]
    ] = None,  # 要处理的列整数索引列表，可选参数，默认为 None
) -> pd.DataFrame:
    """
    对目标DataFrame (df1) 中的缺失值进行填充。

    可以根据提供的第二个DataFrame (df2) 的统计数据进行填充（例如，用训练集的
    统计量填充测试集），或者使用 df1 自身的统计数据进行填充。
    可以指定对所有列或部分列（通过位置索引）进行操作，并为数值型和
    类别型数据分别指定填充策略。

    参数:

    df1 : pd.DataFrame
        需要填充缺失值的目标 DataFrame。函数将操作此 DataFrame 的副本。

    df2 : Optional[pd.DataFrame], 默认值: None
        可选的源 DataFrame：
        用于计算填充缺失值所需的统计量（均值、中位数、众数）。
        如果提供了 df2，则使用 df2 中对应列的统计量来填充 df1 中的缺失值。
        这常用于根据训练集 (df2) 的统计数据来填充测试集 (df1)，以避免数据泄露。
        如果为 None，则使用 df1 自身的统计量来填充其缺失值。

    method_for_numeric : str, 默认值: 'mean'：

        指定用于填充【数值型】列中缺失值的策略。

        接受的值:
            - 'mean': 使用该列的平均值进行填充。
            - 'median': 使用该列的中位数进行填充。

    method_for_category : str, 默认值: 'mode'：

        指定用于填充【类别型或对象型】列中缺失值的策略。

        接受的值:
            - 'mode': 使用该列的众数（出现频率最高的值）进行填充。如果存在多个众数，通常会使用第一个。

    columns : Optional[List[int]], 默认值: None：

        一个可选的整数列表，其中包含要进行缺失值填充的列的【位置索引】(0-based)。
        例如，`[0, 2, 5]` 表示只处理 DataFrame 的第1列、第3列和第6列。
        如果为 None (默认情况)，则函数将尝试处理 df1 中的所有适用列。

    返回:

    pd.DataFrame：

        一个新的 DataFrame，它是 df1 的副本，并且其中指定的缺失值已被填充。

    异常:

    ValueError:

        如果 `method_for_numeric` 或 `method_for_category` 参数的值不是预期的有效字符串。
    """
    Objdf = df1.copy()
    if columns == None:
        columns = list(range(Objdf.shape[1]))
    if df2 is None:
        for column in columns:
            if pd.api.types.is_numeric_dtype(Objdf.iloc[:, column]) == True:
                if method_for_numeric == "mean":
                    fill_value = Objdf.iloc[:, column].mean()
                elif method_for_numeric == "median":
                    fill_value = Objdf.iloc[:, column].median()
                else:
                    raise ValueError(
                        "错误：method_for_numeric参数值无效！应为'mean'或者'median'!"
                    )

                Objdf.iloc[:, column] = Objdf.iloc[:, column].fillna(fill_value)
            else:
                if method_for_category == "mode":
                    fill_value = Objdf.iloc[:, column].mode()
                else:
                    raise ValueError("错误：method_for_category参数值无效！应为'mode'!")
                Objdf.iloc[:, column] = Objdf.iloc[:, column].fillna(fill_value[0])
    else:
        df_anothor = df2.copy()
        for column in columns:
            if pd.api.types.is_numeric_dtype(Objdf.iloc[:, column]) == True:
                if method_for_numeric == "mean":
                    fill_value_numeric = df_anothor.iloc[:, column].mean()
                elif method_for_numeric == "median":
                    fill_value_numeric = df_anothor.iloc[:, column].median()
                else:
                    raise ValueError(
                        "错误：method_for_numeric参数值无效！应为'mean'或者'median'!"
                    )

                Objdf.iloc[:, column] = Objdf.iloc[:, column].fillna(fill_value_numeric)
            else:
                if method_for_category == "mode":
                    fill_value_category = df_anothor.iloc[:, column].mode()
                else:
                    raise ValueError("错误：method_for_category参数值无效！应为'mode'!")
                Objdf.iloc[:, column] = Objdf.iloc[:, column].fillna(
                    fill_value_category[0]
                )
    return Objdf


def onehot_encode(
    data: Union[
        pd.DataFrame, pd.Series, np.ndarray, List
    ],  # data 参数可以接受多种类数组类型
    prefix: Optional[Union[str, List[str], Dict[Any, str]]] = None,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    columns: Optional[List[Any]] = None,  # 列标签可以是任何可哈希类型，不仅仅是str
    sparse: bool = False,
    drop_first: bool = False,
    dtype: Optional[np.dtype] = None,  # 明确指定 NumPy 数据类型
) -> pd.DataFrame:  # pd.get_dummies 通常返回 DataFrame (即使输入是Series)
    """
    对 DataFrame 或 Series 中的分类变量进行独热编码。
    这个函数是 pandas.get_dummies 的一个直接封装，旨在提供一个便捷的调用接口，
    同时保持其所有原始功能和默认行为。

    参数 (与 pandas.get_dummies 的参数含义和默认值一致):
    data (pd.DataFrame, pd.Series, array-like):
        要进行独热编码的数据。
    prefix (str, List[str], Dict, or None, default=None):
        附加到新生成的独热编码列名前面的字符串前缀。
        - 如果为 None: pandas.get_dummies 会根据输入 data 的类型有其默认行为。
            例如，若 data 是 DataFrame 且 columns=None，则用原始列名。
            若 data 是 Series，则可能不加前缀，或用 Series.name (如果存在)。
        - 如果是字符串: 所有进行独热编码的列（由 columns 指定或自动选择）产生的新列都使用此字符串作为统一前缀。
        - 如果是列表: 长度必须与进行独热编码的列数量匹配，按顺序为各列指定前缀。
        - 如果是字典: 键是要进行独热编码的列名，值是对应的前缀。
    prefix_sep (str, default='_'):
        前缀和类别值之间的分隔符。
    dummy_na (bool, default=False):
        是否为 NaN (缺失值) 创建一个单独的指示列。
    columns (Optional[List[Any]], default=None):
        一个列标签的列表，指定 DataFrame 中哪些列需要进行独热编码。
        如果为 None (默认)，并且 data 是 DataFrame，则会尝试对所有数据类型为
        object 或 category 的列进行独热编码。如果 data 是 Series，此参数被忽略。
    sparse (bool, default=False):
        生成的虚拟列是否应该使用稀疏 DataFrame (pd.SparseDataFrame) 表示。
    drop_first (bool, default=False):
        是否删除每个分类特征的第一个类别对应的虚拟列（k-1 独热编码），有助于避免多重共线性。
    dtype (Optional[np.dtype], default=None):
        生成的虚拟列的数据类型。pandas.get_dummies 默认为 np.uint8。

    返回:
    pd.DataFrame:
        进行了独热编码处理后的 DataFrame。

    异常:
    可能会抛出与 pandas.get_dummies 相同的异常，例如输入类型不匹配或参数组合无效。
    """
    df_encoded = pd.get_dummies(
        data=data,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        columns=columns,
        sparse=sparse,
        drop_first=drop_first,
        dtype=dtype,
    )
    return df_encoded


def name_switch_intindex(df: pd.DataFrame, columns_name_list: List) -> List:
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
        intindex = df.columns.get_loc(name)
        intindexList.append(intindex)
    return intindexList
