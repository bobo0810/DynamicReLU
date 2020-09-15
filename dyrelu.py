import torch
import torch.nn as nn
'''
动态ReLU
https://github.com/Islanna/DynamicReLU
'''
class DyReLU(nn.Module):
    '''
    reduction=8 是性能和计算量权衡的选择。  4较比8提升较弱
    k=2 激活函数中分段函数的个数，常规均为2
    '''
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        # 类似SE模块，降维再升维  为ab系数生成对应的残差
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        # register_buffer将tensor注册成buffer，其参数 不进行更新
        # ab值= 初始值 + λ*残差值
        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float()) # λ控制 残差范围
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float()) # a\b系数初始值  a1=1,a2=b1=b2=0,即ReLU.
    def get_relu_coefs(self, x):
        #  '2d'时 为GAP全局平均池化 [2,64,112,112] -> [2,64]
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        # [2,64]-> [2,16] 因为R=4
        theta = self.fc1(theta) # 降维
        theta = self.relu(theta)
        theta = self.fc2(theta) # 升维，得到a、b系数对应的残差
        theta = 2 * self.sigmoid(theta) - 1 # 规范化到-1~1之间
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    '''
    更适合对图像分类等任务
    reduction=8 是性能和速度的权衡
    conv_type='2d' 类似SE模块  '1d'适用语音转文字任务  https://github.com/Islanna/DynamicReLU/issues/2
    '''
    def __init__(self, channels, reduction=8, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels) # 超函数的输出为2KC，覆盖父类的fc2值

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x) # 计算ab值的残差

        # ab值= λ*残差值 + 初始值
        # [2,64,4]
        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1   permute通道交换[2,64,112,112]->[112,112,2,64]
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            # 激活函数 y=ax+b
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            # K=2 表示激活函数内有 两个线性函数，激活其中的最大值
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result
