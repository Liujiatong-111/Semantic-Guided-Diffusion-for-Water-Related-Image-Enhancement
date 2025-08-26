import torch
import torch.nn as nn


class DenseConvLayer(nn.Module):
    """
    单层密集连接卷积模块：包括卷积、激活函数和输入输出的拼接。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DenseConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        # 将输入和输出在通道维度上拼接
        return torch.cat([x, out], dim=1)


class GT_model(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super(GT_model, self).__init__()
        growth_rate = 16  # 每一层增加的通道数

        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channel, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        # 后续卷积层（密集连接）
        self.dense2 = DenseConvLayer(growth_rate, growth_rate)
        self.dense3 = DenseConvLayer(growth_rate * 2, growth_rate)
        self.dense4 = DenseConvLayer(growth_rate * 3, growth_rate)
        self.dense5 = DenseConvLayer(growth_rate * 4, growth_rate)
        self.dense6 = DenseConvLayer(growth_rate * 5, growth_rate)
        self.dense7 = DenseConvLayer(growth_rate * 6, growth_rate)
        self.dense8 = DenseConvLayer(growth_rate * 7, growth_rate)

        # 最后一层，将通道数调整到输出通道数
        self.final_conv = nn.Conv2d(growth_rate * 8, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # 激活函数
        act = nn.LeakyReLU()

        # 前向传播
        x = act(self.conv1(x))  # 第一层
        x = self.dense2(x)  # 第二层
        x = self.dense3(x)  # 第三层
        x = self.dense4(x)  # 第四层
        x = self.dense5(x)  # 第五层
        x = self.dense6(x)  # 第六层
        x = self.dense7(x)  # 第七层
        x = self.dense8(x)  # 第八层

        # 输出层
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    img = torch.randn(1,6,32,32)
    model = GT_model(in_channel=6)
    print(model(img).shape)