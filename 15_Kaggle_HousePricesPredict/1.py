import pandas as pd
import numpy as np
import torch.optim.adam
import Fill_missing_data as Fmd
import torch
from torch import nn
from torch.utils import data
from sklearn.model_selection import train_test_split


def initize_net(layer):
    if type(layer) == nn.Linear:
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.fill_(0)

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
    torch.log(labels)))
    return rmse.item()

# * ----------------------------------------- 数据预处理 ---------------------------------------------

# todo 加载数据
TrainData = pd.read_csv(
    "/home/lighthouse/Myproject/dl1_dlpractice1/15_Kaggle_HousePricesPredict/data/train.csv"
)
TestData = pd.read_csv(
    "/home/lighthouse/Myproject/dl1_dlpractice1/15_Kaggle_HousePricesPredict/data/test.csv"
)
# todo 分离Y标签
Y_totaltrain = TrainData.pop("SalePrice")
# // Y_totaltrain.to_excel("Y_totaltrain.xlsx")

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
X_totaltrain = final_train_data.set_index("Id", inplace=False, drop=True)
# // X_totaltrain.to_excel('X_totaltrain.xlsx')
X_test = final_test_data.set_index("Id", drop=True, inplace=False)
# // X_test.to_excel('X_test.xlsx')

# todo 划分训练集和验证集并转换成dataset类型
X_train, X_val, Y_train, Y_val = train_test_split(
    X_totaltrain, Y_totaltrain, train_size=0.8, test_size=0.2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_ds = torch.tensor(X_train.values, dtype=torch.float32)
Y_train_ds = torch.tensor(Y_train.values, dtype=torch.float32)
X_val_ds = torch.tensor(X_val.values, dtype=torch.float32)
Y_val_ds = torch.tensor(Y_val.values, dtype=torch.float32)
X_totaltrain_ds = torch.tensor(X_totaltrain.values, dtype=torch.float32)
Y_totaltrain_ds = torch.tensor(Y_totaltrain.values, dtype=torch.float32)

Train_dataset = data.TensorDataset(X_train_ds, Y_train_ds)
Val_dataset = data.TensorDataset(X_val_ds, Y_val_ds)
Train_total_dataset = data.TensorDataset(X_totaltrain_ds, Y_totaltrain_ds)
# * ----------------------------------------- 训练多层感知机模型 ---------------------------------------------
# todo 设置全局随机种子
seed = 57

dropout1=0.4
torch.manual_seed(seed)
batch_size = 256
train_iter = data.DataLoader(
    Train_dataset, batch_size=batch_size, shuffle=True, num_workers=6
)
val_iter = data.DataLoader(
    Val_dataset, shuffle=True, batch_size=batch_size, num_workers=6
)
net = nn.Sequential(
    nn.Linear(330,128),
    nn.ReLU(),
    nn.Linear(128,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
loss =nn.MSELoss()
num_epochs = 200
net.apply(initize_net)
trainer = torch.optim.Adam(net.parameters(), lr=0.1 , weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=10, gamma=0.1)
for epoch in range(num_epochs):
    train_ls, test_ls = [], []
    train_mse=0
    val_mse=0
    batchcount=0
    for x, y in train_iter:
        net.train()
        l = loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        with torch.no_grad():
            train_ls.append(log_rmse(net, X_train_ds, Y_train_ds))
            batchcount+=1
            train_mse+=l
    scheduler.step()
    net.eval()
    with torch.no_grad():
        test_ls.append(log_rmse(net, X_val_ds, Y_val_ds))
        val_mse=loss(net(X_val_ds),Y_val_ds)
        print(
            f"epoch:{epoch+1}:训练集均方误差开方：{(train_mse/batchcount)**0.5}，测试集开方：{val_mse**0.5}"
        )
