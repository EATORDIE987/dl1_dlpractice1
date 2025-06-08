import pandas as pd
import numpy as np
import Fill_missing_data as Fmd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    KFold,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import cupy as cp

# * ----------------------------------------- 数据预处理 ---------------------------------------------

# todo 加载数据
TrainData = pd.read_csv(
    "/home/lighthouse/Myproject/dl1_dlpractice1/15_Kaggle_HousePricesPredict/data/train.csv"
)
TestData = pd.read_csv(
    "/home/lighthouse/Myproject/dl1_dlpractice1/15_Kaggle_HousePricesPredict/data/test.csv"
)
# todo 分离Y标签
Y_train = TrainData.pop("SalePrice")
Y_train.to_excel("Y_train.xlsx")

# ! 训练集和测试集的缺失值填充都只能用训练集的统计量！以免发生数据泄露！
# todo 选取训练集的缺失值无意义的列进行填充
columnname_meaningless_traindata = [
    "LotFrontage",
    "MasVnrArea",
    "Electrical",
    "GarageYrBlt",
]
TrainData_filled = Fmd.Fill_meaningless_missing_data(
    TrainData,
    method_for_numeric="mean",
    method_for_category="mode",
    columns=Fmd.name_switch_intindex(
        df=TrainData, columns_name_list=columnname_meaningless_traindata
    ),
)
# // print(Fmd.Count_missing_data(TrainData_filled,Obj='omission'))

# todo 选取测试集的缺失值无意义的列进行填充
columnname_meaningless_testdata = [
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
]
TestData_filled = Fmd.Fill_meaningless_missing_data(
    df1=TestData,
    df2=TrainData,
    method_for_numeric="mean",
    method_for_category="mode",
    columns=Fmd.name_switch_intindex(
        TestData, columns_name_list=columnname_meaningless_testdata
    ),
)
# // print(Fmd.Count_missing_data(TestData_filled, Obj="omission"))

# todo 上下拼接填充无意义数据后的训练集测试集，以保证onehot编码不存在遗漏一个集合里面没有导致最终训练集测试集列数不同的情况
DataSet = pd.concat([TrainData_filled, TestData_filled], axis=0, ignore_index=True)
# // DataSet.to_excel('dataset.xlsx')

# todo 进行onehot编码(缺失值nan是有意义的)
DataSet_encode = Fmd.onehot_encode(data=DataSet, dummy_na=True, dtype="int32")

# todo 重新将onehot编码后的数据集分开为原来的训练集和测试集
# //print(DataSet_encode.shape,TrainData.shape,TestData.shape)
final_train_data = DataSet_encode.iloc[0:1460, :].copy()
final_test_data = DataSet_encode.iloc[1460:, :].copy()
final_test_data.reset_index(drop=True, inplace=True)
# // final_train_data.to_excel('final_train_data.xlsx')
# // final_test_data.to_excel('final_test_data.xlsx')
# // print(final_train_data.shape,final_test_data.shape)

'''
# * ----------------------------------------- 随机森林训练环节 ---------------------------------------------
# todo 定义交叉验证器
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=40)
scoring_metric = "neg_mean_absolute_error"
"""
# todo 定义基础模型
base_model = RandomForestRegressor(
    random_state=42, n_jobs=-1, criterion="squared_error"
)
# todo 网格搜索
params_grid = {
    "n_estimators": [100, 200, 300, 600],
    "max_features": ["sqrt", 0.3, 0.5],  # 尝试不同的策略
    "max_depth": [None, 15, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
}
GridSearch = GridSearchCV(
    estimator=base_model,
    param_grid=params_grid,
    n_jobs=-1,
    cv=cv_splitter,
    scoring=scoring_metric,
)
GridSearch.fit(X=final_train_data, y=Y_train)
model = GridSearch.best_estimator_
print(GridSearch.best_params_, GridSearch.best_score_)
"""
# todo 使用网格搜索确定超参数后建立随机森林模型
model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    criterion="squared_error",
    max_depth=15,
    max_features=0.3,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=600
)
# todo 评估模型5折交叉验证在训练集上的效果
scores = cross_val_score(
    estimator=model,
    X=final_train_data,
    y=Y_train,
    cv=cv_splitter,
    n_jobs=-1,
    scoring=scoring_metric,
)
# todo 输出结果信息
print('网格搜索超参数调优后，随机森林算法5折交叉验证在训练集上的表现：')
print(f"交叉验证评估指标: {scoring_metric}")
print(f"每一折的分数: {scores}")  # scores 是一个包含 K 个分数的 NumPy 数组
print(f"平均分数: {np.mean(scores):.4f}")
print(f"分数的标准差: {np.std(scores):.4f}")
print(
    f"总结：随机森林模型使用 {5}-折交叉验证，得到的平均 {scoring_metric} 为: {np.mean(scores):.4f} (标准差: {np.std(scores):.4f})"
)
print('----------------------------------------------------------------------------------------------------------------------------')
# todo 使用确定超参数的新随机森林模型在完整训练集上进行训练
model.fit(final_train_data,Y_train)
# todo 预测测试集数据并输出表格
Pre_RF=model.predict(X=final_test_data)
pd.Series(Pre_RF).to_excel('Pre_RF.xlsx')
'''

# * ----------------------------------------- Xgboost训练环节 ---------------------------------------------
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=40)
scoring_metric = 'neg_root_mean_squared_error'
'''
# todo 定义基础xgboost模型
base_model = xgb.XGBRegressor(
    objective="reg:squarederror",  # 回归任务的目标函数，预测平方误差
    random_state=40,  # 随机种子
    #device='cuda',
    #tree_method="hist",
    n_jobs=-1,
)
# todo 网格搜索xgboost模型最佳超参数
param_grid = {
    'n_estimators': [200,300,500],           # 减少组合数量以便快速演示
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5,7],
    'min_child_weight': [1, 3],
    'subsample': [0.7,0.8, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'gamma': [0, 0.1]               # 节点分裂的最小损失降低
}
GridSearch = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=scoring_metric,
    n_jobs=-1,
    cv=cv_splitter,
)
GridSearch.fit(X=final_train_data,y=Y_train)
best_model=GridSearch.best_estimator_
print(GridSearch.best_params_,GridSearch.best_score_)
'''
# todo 定义交叉验证网格搜索确定最优参数的模型
model=xgb.XGBRegressor(
    colsample_bytree=0.9,
    gamma=0,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=1,
    n_estimators=500,
    subsample=0.7,
    objective="reg:squarederror",  # 回归任务的目标函数，预测平方误差
    random_state=40,  # 随机种子
    n_jobs=-1,
)
# todo 计算训练集误差数据
scores=cross_val_score(
    estimator=model,
    X=final_train_data,
    y=Y_train,
    scoring=scoring_metric,
    cv=cv_splitter,
    n_jobs=-1,
)

# todo 输出结果信息
print('网格搜索超参数调优后，Xgboost算法5折交叉验证在训练集上的表现：')
print(f"交叉验证评估指标: {scoring_metric}")
print(f"每一折的分数: {scores}")  # scores 是一个包含 K 个分数的 NumPy 数组
print(f"平均分数: {np.mean(scores):.4f}")
print(f"分数的标准差: {np.std(scores):.4f}")
print(
    f"总结：Xgboost模型使用 {5}-折交叉验证，得到的平均 {scoring_metric} 为: {np.mean(scores):.4f} (标准差: {np.std(scores):.4f})"
)
print('----------------------------------------------------------------------------------------------------------------------------')
model.fit(final_train_data,Y_train)
Pre_Xgb=model.predict(final_test_data)
pd.Series(Pre_Xgb,name='SalePrice',index=final_test_data['Id']).to_excel('Pre_Xgb.xlsx')
pd.Series(Pre_Xgb,name='SalePrice',index=final_test_data['Id']).to_csv('Pre_Xgb.csv')