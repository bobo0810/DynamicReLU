# [DynamicReLU](https://github.com/Islanna/DynamicReLU)
#### 该仓库收录于[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

# 说明
- ECCV2020收录
- 仅加注释,便于理解

# 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.6        | 1.6.0       | Ubuntu |

### 基本结构
![](https://github.com/bobo0810/DynamicReLU/blob/master/dyrelu.png)

## Example
```
import torch.nn as nn
from dyrelu import DyReluB

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.relu = DyReLUB(10, conv_type='2d')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
```