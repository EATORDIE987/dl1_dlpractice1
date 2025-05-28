import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn


# * ------------------------------------ 随机生成数据集函数 ------------------------------------
# todo 定义生成合成数据的函数
def synthetic_data(w, b, num_examples):
    # todo 生成y=wx+b+噪声的数据集
    # ? 使用torch.normal生成正态分布数据X。
    # ? X的每个元素服从均值为0、标准差为1的正态分布。
    # ? X的形状是 (num_examples, len(w))，
    # ? 其中num_examples是样本数量（代表生成了多少数据），len(w)是特征的数量（即权重向量w的长度，也就是向量X的变量x个数）。
    X = torch.normal(0, 1, (num_examples, len(w)))

    # todo 根据线性模型 y = Xw + b 计算真实的标签Y（不含噪声部分）。
    # ? torch.matmul 执行矩阵乘法：X与w相乘，计算每组X的数据对应的y值
    Y = torch.matmul(X, w) + b

    # todo 为标签Y添加噪声。
    # ? 噪声也服从均值为0、标准差为0.01的正态分布。
    # ? 噪声的形状与Y相同，模拟真实数据中的随机误差。
    Y += torch.normal(0, 0.01, Y.shape)

    # todo 返回特征X和重塑后的标签Y。
    # ? Y.reshape((-1, 1)) 将Y重塑为列向量，确保其形状为 (样本数, 1)，-1代表自行计算所需维度
    # 这在深度学习框架中通常是标签期望的格式。
    return X, Y.reshape((-1, 1))


# * ------------------------------------ 生成batch数据集迭代器函数 ------------------------------------
# todo
def load_array(data_array, batch_size, is_train=True):
    # todo 将特征和标签数据封装成PyTorch的DataLoader。
    # ? 函数参数:
    # ?     data_array (tuple): 包含特征张量和标签张量的元组 (features, labels)。
    # ?     batch_size (int): 每个批次的大小。
    # ?     is_train (bool): 是否为训练模式。如果是，则打乱数据。
    # ? 返回:
    # ?     torch.utils.data.DataLoader: PyTorch数据加载器。

    # ? 1. *data_array (星号操作符 - 参数解包)
    # ?     在 Python 中，当星号 * 用在函数调用的参数前面时，它执行的是参数解包的操作。
    # ?     如果 data_array 是一个元组 (A, B)，那么 data.TensorDataset(*data_array) 就等同于 data.TensorDataset(A, B)。
    # ?     如果 data_array 是一个元组 (A, B, C)，那么 data.TensorDataset(*data_array) 就等同于 data.TensorDataset(A, B, C)。
    # ?     在本例中，因为 data_array 是 (features, labels)，所以 data.TensorDataset(*data_array) 实际上被 Python 解释为：data.TensorDataset(features, labels)
    # ? 2. data.TensorDataset
    # ?     torch.utils.data.TensorDataset 是 PyTorch 提供的一个工具类，用于将一个或多个具有相同第一维度大小的张量包装成一个数据集。
    # ?     输入： 它的构造函数接受一个或多个张量作为参数。关键要求是：所有传入的张量的第一个维度必须相同。 这个第一个维度通常代表数据集中的样本数量。
    # ?     在你的例子中，features 的形状是 (1000, 2)，labels 的形状是 (1000, 1)。它们的第一维度都是 1000（样本数量），所以满足这个要求。
    # ?     功能：
    # ?         封装数据： 它将这些输入的张量（例如特征和标签）“绑定”在一起。
    # ?         实现 Dataset 接口： TensorDataset 是 torch.utils.data.Dataset 的一个子类。这意味着它实现了两个必要的方法：
    # ?             __len__(): 当你调用 len(dataset) 时，它会返回数据集中样本的数量（即输入张量的第一个维度的大小，这里是 1000）。
    # ?             __getitem__(idx): 当你使用索引访问数据集时，例如 dataset[i]，它会从每个封装的张量中取出第 i 个样本（即第 i 行数据），并将它们作为一个元组返回。
    # ?     创建 TensorDataset 类型的主要目的是为了能够方便地与 torch.utils.data.DataLoader 一起使用。DataLoader 需要一个 Dataset 对象作为输入，
    # ?     以便进行数据批处理(batching)、打乱顺序 (shuffling) 和并行加载等操作，这些都是训练深度学习模型时非常常见的需求。
    dataset = data.TensorDataset(*data_array)
    # ? 1. data.DataLoader 类
    # ?     DataLoader 的主要职责是：
    # ?         (i)     数据批处理 (Batching): 将数据集分割成小批次。模型通常不是一次处理整个数据集，也不是一次处理一个样本，而是按批次处理。
    # ?         (ii)    数据打乱 (Shuffling): 在每个训练周期 (epoch) 开始时，可以随机打乱数据的顺序，这有助于提高模型的泛化能力，防止模型学习到数据的特定顺序。
    # ?         (iii)   并行加载 (Parallel Loading): 可以使用多个子进程来并行加载数据，这可以显著加快数据准备的速度，尤其是在数据预处理比较复杂或 I/O 成为瓶颈时。
    # ?         (iv)    内存效率: 按批次加载数据，而不是一次性将整个数据集加载到内存中，这对于大型数据集至关重要。
    # ?         (v)     它返回一个可迭代对象 (iterable)，你可以像遍历列表一样遍历它来获取数据批次。
    # ? 2. 参数详解
    # ?     a. dataset
    # ?         类型： torch.utils.data.Dataset 的实例。
    # ?         作用： 这是 DataLoader 的数据源。DataLoader 从这个 dataset 对象中提取数据样本。
    # ?         DataLoader 内部会调用 dataset.__len__() 来确定总共有多少数据，并多次调用 dataset.__getitem__(idx) 来获取单个数据样本，然后将这些样本组合成批次。
    # ?     b. batch_size
    # ?         类型： 整数 (int)。
    # ?         作用： 定义了每个批次中包含的样本数量
    # ?     c. shuffle=is_train
    # ?         shuffle 参数类型： 布尔值 (bool)。
    # ?         作用： 决定是否在每个周期 (epoch) 开始前打乱数据的顺序
    # ?     d. num_workers (int, default=0): 设置用于数据加载的子进程数量。
    # ?         0 表示数据将在主进程中加载。
    # ?         大于 0 的值会启动指定数量的子进程来并行加载数据，这可以显著加快数据准备速度，防止 CPU 成为瓶颈，让 GPU 保持忙碌。
    # ?     e. pin_memory (bool, default=False):
    # ?         如果为 True，并且你正在使用 CUDA (GPU)，DataLoader 会将加载的张量复制到 CUDA 固定内存 (pinned memory) 中。
    # ?         这可以加速数据从 CPU 到 GPU 的传输。通常在 num_workers > 0 时效果更佳。
    # ?     f. drop_last (bool, default=False):
    # ?         如果数据集中的样本总数不能被 batch_size 整除，那么最后一个批次可能会比 batch_size 小。
    # ?         如果 drop_last=True，则这个最后的不完整批次将被丢弃。
    # ?         如果 drop_last=False，则最后一个批次会包含剩余的样本，因此它的大小可能会小于 batch_size。
    # ? 3. DataLoader 返回什么？
    # ?     data.DataLoader(...) 的调用会返回一个 DataLoader 实例。
    # ?     这个实例是一个可迭代对象。你可以使用 for 循环来遍历它，或者像本代码中那样，使用 iter() 和 next() 来手动获取批次。
    # ?     当遍历 DataLoader 时，它在每次迭代中会产生 (yield) 一个数据批次。这个批次的结构通常是一个列表或元组，
    # ?     其中包含了从 dataset.__getitem__ 返回的各项内容的批处理版本。
    # ?     对于 TensorDataset，如果 dataset[i] 返回 (feature_i, label_i)，那么 DataLoader 产生的一个批次通常是 [batch_of_features, batch_of_labels]：
    # ?     batch_of_features: 一个张量，形状为 (batch_size, num_features)。
    # ?     batch_of_labels: 一个张量，形状为 (batch_size, num_label_dims)。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# * ------------------------------------ 随机生成数据集 ------------------------------------
# todo 定义真实的权重w和偏置b，这些是我们要模拟的线性模型的参数
# 真实的权重向量，有两个特征对应的权重
true_w = torch.tensor([2, -3.4])
# 真实的偏置项
true_b = 4.2

# todo 调用synthetic_data函数生成包含1000个样本的合成数据集
# features 存储生成的特征X，labels 存储生成的标签Y
features, labels = synthetic_data(true_w, true_b, 1000)

# * ------------------------------------ 绘图代码 ------------------------------------
"""
# ? 创建一个新的图形，并设置其大小为8x6英寸
plt.figure(figsize=(8, 6))

# todo 绘制散点图：只选择特征X的第二个维度（索引为1）与标签Y的关系
# ? features[:, 1] 选择所有样本的第二个特征。
# ? .detach() 将张量从计算图中分离，使其不再追踪梯度，这是进行Numpy转换前的最佳实践。
# ? .numpy() 将PyTorch张量转换为NumPy数组，Matplotlib绘图函数通常需要Numpy数组作为输入。
# ? s=10 设置散点的大小。
# ? alpha=0.7 设置散点的透明度，0表示完全透明，1表示完全不透明，用于处理大量重叠点。
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=10, alpha=0.7)

# ? 添加网格线，帮助查看数据点的分布，linestyle='--' 设置虚线，alpha=0.6 设置透明度
plt.grid(True, linestyle="--", alpha=0.6)
# ? 显示绘制的图表
plt.show()
"""

# * ------------------------------------ 生成batch迭代器 ------------------------------------
batch_size = 10

data_iter = load_array((features, labels), batch_size)
# //print(next(iter(data_iter)))
# * ------------------------------------ 初始化模型参数 ------------------------------------
# todo 定义一个神经网络模型 'net'。
# ? nn.Sequential: 是一个有序的容器（Sequential Container）,模块将按照在构造函数中传递的顺序添加到容器中。在这里，它只包含一个层。
# ? nn.Linear(2, 1): 定义一个线性层（也叫全连接层或密集层）。
# ?     - 第一个参数 '2': in_features，表示输入特征的数量。这意味着该层期望接收有2个特征的输入样本。
# ?     - 第二个参数 '1': out_features，表示输出特征的数量。这意味着该层将产生有1个特征的输出（即网络有一个输出神经元）。
# ?     - 这个线性层会自动创建权重 (weight) 和偏置 (bias) 参数。
# ?     权重矩阵的形状会是 (out_features, in_features)，即 (1, 2)。
# ?     偏置向量的形状会是 (out_features)，即 (1)。
net = nn.Sequential(nn.Linear(2, 1))
# todo 初始化网络中第一个层（即我们刚刚定义的线性层）的权重参数。
# ? net[0]: 从 nn.Sequential 容器中获取第一个模块（层）。
# ?         因为我们只定义了一个 nn.Linear(2, 1) 层，所以 net[0] 就是这个线性层。
# ? .weight: 访问该线性层的权重参数。这是一个 torch.nn.Parameter 对象，
# ?          其内部包含一个张量 (tensor)。
# ? .data:   直接访问权重参数内部的实际数据张量 (torch.Tensor)。
# ?          这样做是为了在不被 PyTorch 的自动求导机制追踪的情况下修改权重值。
# ?          这在初始化权重时是常见的做法。
# ? .normal_(0, 0.01):
# !          这是一个原地 (in-place) 操作（由末尾的下划线 '_' 表示）。
# ?          它会用从正态分布（高斯分布）中随机抽取的值来填充权重张量。
# ?    - 第一个参数 '0': 正态分布的均值 (mean)。
# ?    - 第二个参数 '0.01': 正态分布的标准差 (standard deviation)。
# ?    - 所以，权重会被初始化为均值为0，标准差为0.01的随机数。
net[0].weight.data.normal_(0, 0.01)
# todo 初始化网络中第一个层（线性层）的偏置参数。
# ? net[0]: 同样是获取 nn.Sequential 中的第一个模块（线性层）。
# ? .bias:   访问该线性层的偏置参数。这也是一个 torch.nn.Parameter 对象。
# ? .data:   直接访问偏置参数内部的实际数据张量。
# ? .fill_(0): 这是一个原地操作。
# ?            它会用指定的值来填充偏置张量。
# ?    - 参数 '0': 指定填充的值。
# ?    - 所以，偏置参数的所有元素都被初始化为0。
net[0].bias.data.fill_(0)

# * ------------------------------------ 定义损失函数MSE ------------------------------------
loss = nn.MSELoss()

# * ------------------------------------ 小批量随机梯度下降（SGD） ------------------------------------
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# * ------------------------------------ 训练模型 ------------------------------------
# todo 设置训练的总轮数 (number of epochs)
num_epoch = 3
# todo 开始外层循环，遍历指定的训练轮数。
for epoch in range(num_epoch):
    # todo 开始内层循环，遍历数据加载器 'data_iter' 中的所有批次 (batch)。
    for x, y in data_iter:
        # ? 1. 前向传播：计算模型对当前批次特征 'x' 的预测输出。net(x) 会调用模型 'net' 的 forward 方法。
        # ? 2. 计算损失：使用之前定义的 'loss' 函数 (MSELoss) 计算预测输出 net(x) 与真实标签 'y' 之间的损失值。变量 'l' 存储了这个批次的损失。
        l = loss(net(x), y)
        # ? 3. 清零梯度：在进行反向传播之前，必须清除先前计算的梯度。
        # ?             这是因为 PyTorch 默认会累积梯度，如果不清零，当前批次的梯度会叠加到之前的梯度上，导致错误。
        # ?             trainer.zero_grad() 会将模型中所有可训练参数的 .grad 属性设置为零。
        trainer.zero_grad()
        # ? 4. 反向传播：根据当前批次的损失 'l' 计算模型参数的梯度。
        # ?             l.backward() 会自动计算损失 'l' 相对于模型所有可训练参数的偏导数（梯度）。
        # ?             这些梯度值会存储在参数的 .grad 属性中。
        # ? 这里l是已经计算好的，不需要加入.sum()了，它已经是一个标量了
        l.backward()
        # ? 5. 更新参数：优化器根据计算出的梯度来更新模型的参数。
        # ?             trainer.step() 会根据在实例化 'trainer' 时选择的优化算法 (SGD) 和学习率 (lr=0.03) 来调整参数。
        # ?             例如，对于 SGD，参数更新规则通常是：parameter = parameter - learning_rate * parameter.grad。
        # ! 操作不要遗漏双括号！！！
        trainer.step()
    # todo 在每个 epoch 结束后，计算整个训练集 (features, labels) 上的损失。
    # 这有助于了解模型在整个数据集上的整体表现，而不仅仅是最后一个批次的表现。
    # ! 注意：这只是评估损失，并不会用于梯度更新，因为梯度清零和更新步骤在批次循环内。
    l = loss(net(features), labels)
    print(f"epoch{epoch+1},loss{l:f}")
