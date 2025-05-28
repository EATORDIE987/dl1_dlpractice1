import torch
import random
import matplotlib.pyplot as plt


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


# * ------------------------------------ batch函数 ------------------------------------
# todo 定义一个数据迭代器函数，用于生成小批量（mini-batch）数据
# ! 这在训练神经网络时非常常用，可以减少内存消耗并加速训练
def data_iter(batch_size, features, labels):
    """
    生成特征和标签的迭代器，每次返回一个随机的小批量数据。

    参数:
        batch_size (int): 每个小批量的样本数量。
        features (torch.Tensor): 包含所有特征的张量。
        labels (torch.Tensor): 包含所有标签的张量。

    生成器:
        yield (torch.Tensor, torch.Tensor): 每次迭代返回一个包含特征和标签的小批量张量对。
    """
    # ? len()返回tensor的第一个维度的长度
    # todo 获取样本总数量
    num_examples = len(features)

    # todo 创建一个包含所有样本索引的列表，例如 [0, 1, 2, ..., num_examples-1]
    indice = list(range(num_examples))

    # ? 使用random.shuffle对索引列表进行随机打乱。
    # ! 这样做是为了确保每个训练周期（epoch）中，数据都是随机顺序的，
    # ! 避免模型学习到数据固有的顺序性，提高泛化能力。
    random.shuffle(indice)

    # todo 遍历整个数据集，每次步长为batch_size，以生成小批量数据集
    for i in range(0, num_examples, batch_size):
        # todo 计算当前批次的结束索引，确保不会超出总样本数量
        # ? min(i + batch_size, num_examples) 用于处理最后一个批次可能不足batch_size的情况
        batch_end_index = min(i + batch_size, num_examples)

        # todo 从打乱的索引列表中取出当前批次的索引
        # ? 使用torch.tensor将其转换为PyTorch张量，以便后续用于张量索引
        batch_indice = torch.tensor(indice[i:batch_end_index])

        # todo 使用这些批次索引从features和labels中抽取对应的小批量数据
        # ?并使用yield关键字将其返回。yield使得这个函数成为一个生成器，
        # ?每次调用时才计算并返回一个批次，而不是一次性加载所有数据到内存。
        yield features[batch_indice], labels[batch_indice]


# * ------------------------------------ 定义线性回归模型 ------------------------------------
# todo 定义线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# * ------------------------------------ 定义损失函数（MSE) ------------------------------------
# todo 定义MSE为损失函数（这里还没有平均）
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# * ------------------------------------ 小批量梯度下降（SGD) ------------------------------------
# todo 执行小批量随机梯度下降(SGD)更新。
def sgd(params, lr, batch_size):
    # ? params (list): 包含所有需要更新的模型参数（例如，w和b）的列表。这些参数通常是PyTorch张量。
    # ? lr (float): 学习率（learning rate），控制每次参数更新的步长大小。
    # ? batch_size (int): 当前小批量的样本数量。梯度的平均值会除以这个批次大小。

    # ? torch.no_grad() 是一个上下文管理器。
    # ? 在这个with块内部执行的所有操作都不会被PyTorch记录到计算图中，
    # ? 也就是说，不会计算梯度。
    # ? 这是为了确保参数更新（param -= ...）操作本身不会产生新的梯度信息，
    # ? 从而避免在更新参数时对参数的梯度求梯度。
    with torch.no_grad():
        # ? 遍历params列表中的每一个参数（例如，权重w和偏置b）
        for param in params:
            # ? 更新参数并除以batch_size以计算MSE的平均
            param -= lr * param.grad / batch_size
            # ? 将当前参数的梯度清零。
            # ? 这是非常重要的一步！PyTorch会累积梯度，
            # ? 如果不清零，下一次反向传播计算出的梯度会与之前的梯度累加，导致错误的结果。
            param.grad.zero_()


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
# * ------------------------------------ batch ------------------------------------
# todo 定义每个小批量的样本数量为10
batch_size = 10

"""
# todo 使用之前定义的 data_iter 函数来遍历数据集。
# ? data_iter 会根据指定的 batch_size 和数据（features, labels）,每次返回一个随机选择的小批量特征 x 和对应的标签 y。
for x, y in data_iter(batch_size, features, labels):
    print(x, "\n", y)
    # ? 在第一次迭代后立即跳出循环。这样做是为了演示一个批次的数据结构，而不是打印所有批次。
    break
"""

# * ------------------------------------ 定义初始化模型参数 ------------------------------------
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# * ------------------------------------ 模型训练 ------------------------------------
# todo 定义参数训练模型
# ? 定义学习率（learning rate），控制每次参数更新的步长
lr = 0.03
# ? 'net' 变量在这里应该指向线性回归模型函数（即前向传播函数）
net = linreg
# ? 定义训练的周期数（epoch），即遍历整个数据集的次数
num_epoch = 3
# ? 'loss' 变量指向定义的损失函数，这里是均方误差
loss = squared_loss

# todo 外层循环：迭代每个训练周期（epoch）
for epoch in range(num_epoch):
    # todo 内层循环：遍历数据集中的每一个小批量数据
    for x, y in data_iter(batch_size, features, labels):
        # 前向传播
        # todo 调用模型函数 net（即 linreg），传入当前批次的特征 x、权重 w 和偏置 b，计算得到模型的预测值。
        # todo 将模型的预测值和真实标签 y 传入损失函数 loss，计算当前批次的损失 l。
        l = loss(net(x, w, b), y)
        # 反向传播：
        # todo 调用 l.sum().backward() 来计算损失 l 对模型参数 w 和 b 的梯度。
        # ? .sum() 是必要的，因为如果 l 是一个非标量张量（例如，每个样本的损失组成的向量），
        # ? backward() 需要一个标量输入来计算梯度。求和操作将其变为标量。
        # ? 这一步会填充 w.grad 和 b.grad 属性。
        l.sum().backward()
        # todo 梯度下降更新参数
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        # todo 计算本轮训练后的loss
        train_l = loss(net(features, w, b), labels)
        print(f"epoch{epoch+1},loss{float(train_l.mean()):f}")
