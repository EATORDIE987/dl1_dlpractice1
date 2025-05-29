import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn


# * ---------------------------------- 定义累加器的类 ----------------------------------
# ! 很重要的累加器，可用于别的程序
class Accumulator:  # @save
    def __init__(self, n):
        """
        # ? 构造函数 (初始化方法)。
        # ? 当创建一个 Accumulator 对象时，这个方法会被调用。
        参数:
        # ? n (int): 需要累加的变量的数量。例如，如果 n=2，则这个累加器可以同时追踪两个数值的累加。
        """
        # ? self.data 是一个实例变量，它被初始化为一个列表。
        # ? 这个列表包含 n 个元素，每个元素都被初始化为浮点数 0.0。
        # ? 这个列表将用来存储各个变量的累加值。
        # ? 例如，如果 Accumulator(3) 被调用，则 self.data 将是 [0.0, 0.0, 0.0]。
        self.data = [0.0] * n

    def add(self, *args):
        # ? zip(self.data, args) 会将 self.data 列表中的元素和传入的 args 元组中的元素
        # ? 一一配对。例如，如果 self.data = [a1, a2] 而 args = (b1, b2)，
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # todo 将所有累加的变量重置为 0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        # ! 使得 Accumulator 对象可以使用方括号索引来获取特定变量的累加值。
        这是 Python 中的一个“魔术方法”(magic method) 或“双下划线方法”(dunder method)。
        参数:
        # ? idx (int): 要获取的累加变量的索引 (从 0 开始)。
        返回:
        # ? float: 索引 idx 对应的变量的当前累加值。
        """
        # ? 直接返回 self.data 列表中索引为 idx 的元素。
        # ? 例如，如果 acc 是一个 Accumulator 对象，acc[0] 就会调用这个方法并返回 self.data[0]。
        return self.data[idx]


# * ---------------------------------- 定义精确率函数 ----------------------------------
# todo 计算精确率
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# * ---------------------------------- 计算在指定数据集上模型的精度 ----------------------------------
# todo 计算在指定数据集上模型的精确率
def evaluate_accuracy(net, data_iter):
    """
    参数:
    # ? net (torch.nn.Module): 需要评估的 PyTorch 模型。
    # ? data_iter (torch.utils.data.DataLoader): 包含数据的数据加载器，它会迭代产生一批批的特征 (X) 和标签 (y)。
    返回:
    # ? float: 模型在该数据集上的整体准确率。
    """

    # ? 1. 检查模型是否为 PyTorch 的 nn.Module 类型
    # ?     这是为了确保我们传入的是一个合法的 PyTorch 模型
    if isinstance(net, torch.nn.Module):
        # ? 2. 将模型设置为评估模式 (evaluation mode)
        # ?     这非常重要，因为它会关闭一些在训练时启用但在评估时应禁用的层，
        # ?     例如 Dropout 层和 BatchNorm 层（在评估时会使用其学到的全局统计量而不是当前批次的统计量）。
        # ?     如果不设置 net.eval()，评估结果可能会不准确或不稳定。
        net.eval()
        # ? 3. 初始化一个累加器 (Accumulator) 对象
        # ?     这个 Accumulator 类（通常由 d2l 库提供或用户自定义）用于累积两个值：
        # ?     metric[0]: 累积正确预测的样本数量
        # ?     metric[1]: 累积总的预测样本数量
        # ?     这里的 Accumulator(2) 表示它内部维护一个长度为 2 的列表或数组来存储这两个累加值。
    metric = Accumulator(2)
    # ? 4. 禁用梯度计算
    # ?     在模型评估阶段，我们不需要计算梯度，因为我们不会进行参数更新（反向传播）。
    # ?     torch.no_grad() 上下文管理器可以临时关闭所有涉及的张量的 requires_grad 属性，
    # ?     从而减少内存消耗并加速计算。
    with torch.no_grad():
        for X, y in data_iter:
            # ? 6. 进行预测并累加结果
            # ?   a. net(X): 将当前批次的特征 X 输入到模型 net 中，得到模型的预测输出。对于分类问题，这通常是每个类别的原始分数 (logits) 或概率。
            # ?   b. y.numel():计算真实标签张量 y 中元素的总数量。在一个批次中，这通常等于该批次的样本数量。
            # ?   c. metric.add(num_correct_in_batch, num_samples_in_batch):
            # ?      将当前批次的正确预测数和样本总数添加到累加器 metric 中。
            # ?      metric[0] 会加上 num_correct_in_batch
            # ?      metric[1] 会加上 num_samples_in_batch
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# * ---------------------------------- 下载并提取数据 ----------------------------------
# todo 定义一个图像转换操作：将图像数据从 PIL (Python Imaging Library) 类型或者 NumPy 数组类型变换成 PyTorch 的张量 (Tensor) 类型，并且是 32 位浮点数格式。
# todo 同时，这个转换会自动将像素值从 [0, 255] 的范围缩放到 [0, 1] 的范围 (通过除以 255.0)。
trans = transforms.ToTensor()
# todo 加载 FashionMNIST 训练数据集
mnist_train = torchvision.datasets.FashionMNIST(
    # ? root: 指定数据集下载和存放的根目录路径。
    root="/home/lighthouse/Myproject/dl1_dlpractice1/09_SoftmaxRegression/data",
    # ? train=True: 指定加载的是训练集。
    # ? FashionMNIST 数据集分为训练集和测试集。
    train=True,
    # ? transform=trans: 指定对加载的每一张图像应用的转换操作。
    # ? 在这里，就是我们前面定义的 trans (transforms.ToTensor())，
    # ? 它会将图像转换为张量并进行归一化。
    transform=trans,
    # ? download=True: 如果指定路径下不存在数据集文件，则自动从互联网下载。
    # ? 如果已经下载过，则不会重复下载。
    download=True,
)
# todo 加载 FashionMNIST 测试数据集
mnist_test = torchvision.datasets.FashionMNIST(
    root="/home/lighthouse/Myproject/dl1_dlpractice1/09_SoftmaxRegression/data",
    # ? train=False: 指定加载的是测试集。其余参数同上
    train=False,
    transform=trans,
    download=True,
)
# ? mnist_train[0]:
# ? 这表示获取 mnist_train 数据集中的第一个元素（索引为 0 的样本）。
# ? 对于 torchvision 中的图像数据集（如 FashionMNIST），当应用了 transform 时（这里是 transforms.ToTensor()），每个元素通常返回一个元组 (image, label)。
# ? 所以，mnist_train[0] 会返回一个包含两项的元组：(第一张图像的张量, 第一张图像对应的标签)。
# ? mnist_train[0][0]:
# ? 这从上面返回的元组 (image, label) 中取出第一个元素，也就是第一张图像的张量
# // print(mnist_train[0][0].shape)

# * ---------------------------------- 把数据加载进dataset ----------------------------------
# todo 批量的数目
batch_size = 256
# todo 把训练集和测试集加载进dataloader
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

# * ---------------------------------- 初始化模型参数 ----------------------------------
# todo 设置输入输出向量维度
num_inputs = 784
num_outputs = 10
# ? 定义了一个展平层以把图片形式的矩阵data拉伸成向量参与后面的线性层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# todo 定义线性层参数初始值
net[1].weight.data.normal_(0, 0.01)
net[1].bias.data.fill_(0)
# todo 定义交叉熵loss，对每条样本求交叉熵
# ! 此处pytorch为确保交叉熵的数值稳定性采取了 LogSumExp 技巧
loss = nn.CrossEntropyLoss(reduction="none")
lr = 0.05
num_epochs = 10

# * ---------------------------------- 训练模型并评估测试集精确率 ----------------------------------
trainer = torch.optim.SGD(net.parameters(), lr)
# todo 学习率衰减策略
# ? 示例：每4个epoch，lr变为原来的0.1倍
scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=4, gamma=0.2)
for epoch in range(num_epochs):
    # ! 及时切换到训练模式
    net.train()
    for x, y in train_iter:
        l = loss(net(x), y)
        # ! 反向传播前清零梯度
        trainer.zero_grad()
        l.sum().backward()
        # ? 更新参数
        trainer.step()
    # ! 每轮epoch结束更新学习率
    scheduler.step()
    # todo 输出当前测试集预测精准率
    print(f"epoch{epoch+1}:Accuracy:{evaluate_accuracy(net, test_iter)}", "\n")
