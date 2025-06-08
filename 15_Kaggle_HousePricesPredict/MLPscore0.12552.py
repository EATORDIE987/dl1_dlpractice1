import pandas as pd
import numpy as np
import torch.optim.adam
import torch.optim.adam
import Fill_missing_data as Fmd
import torch
from torch import nn
from torch.utils import data
from sklearn.model_selection import train_test_split
import DPAFEF


def initize_net(layer):
    if type(layer) == nn.Linear:
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.fill_(0)


# * ----------------------------------------- 数据预处理 ---------------------------------------------

# todo 加载数据
TrainData = pd.read_csv(
    "/home/lighthouse/Myproject/dl1_dlpractice1/15_Kaggle_HousePricesPredict/data/train.csv",
    index_col="Id",
)
TestData = pd.read_csv(
    "/home/lighthouse/Myproject/dl1_dlpractice1/15_Kaggle_HousePricesPredict/data/test.csv",
    index_col="Id",
)
# print(TrainData)
# print(DPAFEF.Count_missing_data(TrainData, Obj="omission"))
# print(DPAFEF.Count_missing_data(TestData, Obj="omission"))
delete_info, train_data_DelSomeFeatures, test_data_DelSomeFeatures = (
    DPAFEF.Delete_HighPercent_MissingData(TrainData, TestData, alpha=0.4)
)
# print(delete_info, "\n", train_data_DelSomeFeatures, "\n", test_data_DelSomeFeatures)
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

train_data_fillna, test_data_fillna = DPAFEF.Fill_meaningless_misssingdata(
    train_data_DelSomeFeatures,
    test_data_DelSomeFeatures,
    method_numeric="median",
    method_category="mode",
    columns=DPAFEF.nameindex_switchto_intindex(
        train_data_DelSomeFeatures, columnname_meaningless_data
    ),
)
# print(train_data_fillna, "\n", test_data_fillna)
"""
print(
    DPAFEF.Count_missing_data(train_data_fillna),
    "\n",
    DPAFEF.Count_missing_data(test_data_fillna),
)
"""
train_data_final, test_data_final, sigma, mu = DPAFEF.OneHot_Encode(
    train_data_fillna,
    test_data_fillna,
    numeric_method="standardize",
    keep_target_separate=False,
)
# print(train_data_final,'\n',test_data_final,'\n')
"""
print(
    DPAFEF.Count_missing_data(train_data_final),
    "\n",
    DPAFEF.Count_missing_data(test_data_final),
)
"""


Y = train_data_final.pop("SalePrice")

X_train, X_val, Y_train, Y_val = train_test_split(
    train_data_final, Y, train_size=0.8, random_state=40, shuffle=True, test_size=0.2
)
# print(X_train, X_val, Y_train, Y_val)
torch.manual_seed(57)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将要使用的设备是: {device}")
sigma_tensor = torch.tensor(sigma, device=device)
mu_tensor = torch.tensor(mu, device=device)
batch_size = 64
batch_size_val = 64
train_dataset = data.TensorDataset(
    torch.tensor(X_train.values, dtype=torch.float32),
    torch.tensor(Y_train.values, dtype=torch.float32),
)
val_dataset = data.TensorDataset(
    torch.tensor(X_val.values, dtype=torch.float32),
    torch.tensor(Y_val.values, dtype=torch.float32),
)

train_iter = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=6
)
val_iter = data.DataLoader(
    val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=6
)

net = nn.Sequential(
    nn.Linear(275, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
net.apply(initize_net)
net.to(device)
loss = nn.MSELoss(reduction="mean")
num_epochs = 128
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=80, gamma=0.7)
for epoch in range(num_epochs):
    # --- 训练阶段 ---
    net.train()  # 在epoch开始时设置

    # 用于累加在原始尺度上的“平方和误差” (Sum of Squared Errors)
    total_sse_train = 0.0

    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)
        net.train()
        # 修正形状以匹配模型输出
        y = y.unsqueeze(1)
        trainer.zero_grad()
        y_pred = net(x)
        l = loss(y_pred, y)
        l.backward()
        trainer.step()
        # --- 计算用于报告的指标 (在原始尺度上) ---
        with torch.no_grad():  # 在no_grad下进行，节省计算
            # 恢复到原始尺度
            y_pred_orig = y_pred * sigma_tensor + mu_tensor
            y_true_orig = y * sigma_tensor + mu_tensor

            # 计算批次的平方和误差，并用.item()累加
            total_sse_train += torch.sum((y_pred_orig - y_true_orig) ** 2).item()
    train_rmse = (total_sse_train / len(train_dataset)) ** 0.5
    scheduler.step()
    # --- 验证阶段 ---
    net.eval()
    total_sse_val = 0.0
    with torch.no_grad():
        for x_val, y_val in val_iter:
            # 修正形状
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_val = y_val.unsqueeze(1)
            y_pred_val = net(x_val)
            # 恢复到原始尺度
            y_pred_orig_val = y_pred_val * sigma_tensor + mu_tensor
            y_true_orig_val = y_val * sigma_tensor + mu_tensor
            # 累加验证集的平方和误差
            total_sse_val += torch.sum((y_pred_orig_val - y_true_orig_val) ** 2).item()
    # 计算整个验证集的RMSE
    val_rmse = (total_sse_val / len(val_dataset)) ** 0.5
    print(f"epoch:{epoch+1}: 训练集RMSE: {train_rmse:.4f}, 验证集RMSE: {val_rmse:.4f}")


# ===================================================================
#                      测试集预测与提交
# ===================================================================

# 1. 准备测试数据张量，并移动到设备
# test_data_final 是你预处理好的测试集 DataFrame
test_tensor = torch.tensor(test_data_final.values, dtype=torch.float32).to(device)

# 2. 将模型设置为评估模式
net.eval()

# 3. 在 no_grad 上下文中进行预测，不计算梯度
with torch.no_grad():
    # 得到标准化的预测值
    predictions_std = net(test_tensor)

    # 4. 恢复预测值到原始尺度
    predictions_orig = predictions_std * sigma_tensor + mu_tensor

# 5. 将结果从GPU移回CPU，并转换为NumPy数组
#    .cpu() - 从GPU移至CPU
#    .numpy() - 从Tensor转换为NumPy array
predictions_numpy = predictions_orig.cpu().numpy()

# 6. 创建用于提交的DataFrame
#    确保使用原始测试集TestData的索引作为'Id'
submission_df = pd.DataFrame(
    {
        "Id": TestData.index,
        # .flatten() 将 [[v1], [v2], ...] 的二维数组展平成 [v1, v2, ...] 的一维数组
        "SalePrice": predictions_numpy.flatten(),
    }
)

# 7. 保存为CSV文件
#    index=False 表示不将DataFrame的索引写入文件，这是Kaggle的要求
submission_df.to_csv("submission.csv", index=False)

print("\n预测完成！提交文件 'submission.csv' 已生成。")
print("提交文件预览:")
print(submission_df.head())
