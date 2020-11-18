# 问题描述
pytorch转caffe后上板，由于caffe仅支持最高思维input，且板子上框架要求网络中所有数据第一维必须为1，而pytorch模型输入尺寸为6\*2\*96\*96，所以在将输入reshape(view)为1\*12\*96\*96后实现转换。

但这样会导致输入被切分为12份分别送入模型，增加了很多运算量。所以在不方便进行模型重训的情况下，使用group convolution的思想来将原本模型的输入从1\*1\*96\*96转换为1\*6\*96\*96或1\*12\*96\*96。

# 方法描述
通过在对pytorch代码(调整nn.conv2d输入通道数以及group参数)以及模型的修改(see below)，实现输入图像在通道上的拼接以及模型输入尺寸的增加。

例：将六个单通道卷积层合并为一个（包括BN，ReLU等）

```python
# Changes in model file (.pth)
model_dict = model.state_dict()
new_state_dict = {}
for key, value in state_dict.items():
    # if prefix is 'unet', the layer needs to be stacked up
    if key.split('.')[0] == 'unet':
        # if dimension is over 2, therefore it is conv layer
        if value.ndim > 1:
            tmp_value = []
            for i in range(6):
                tmp_value.append(value)
            new_state_dict[key] = torch.cat(tmp_value, dim=0)
        # otherwise it is a common layer (e.g. BatchNorm)
        elif value.ndim == 1:
            tmp_value = []
            for i in range(6):
                tmp_value += value.numpy().tolist()
            new_state_dict[key] = torch.from_numpy(np.array(tmp_value))
        # or it is an integer
        else:
            new_state_dict[key] = value * 6
    else:
        new_state_dict[key] = value
model_dict.update(new_state_dict)
model.load_state_dict(model_dict)
```

也可以通过以上方法对模型中的层进行删减（即在循环中跳过不需要的层），**<u>注意修改模型代码</u>**（see below）。

注意 **'group_size'** 在 '**GroupDoubleConv**' 和 '**GroupEncoderDecoder**' 中的用法，对应的需要修改模型输入和输出的尺度，确保输出在通道上能够被group_size整除，且与之前模型文件中的参数维度匹配。

```python
group_size = 6

class GroupDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        ...

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=group_size),
            ...
            
    def forward(self, x):
        ...

class GroupDown(nn.Module):
    ...

class GroupEncoderDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, deploy=False):
        super().__init__()

        self.inc = GroupDoubleConv(n_channels, 16 * group_size)
        self.down1 = GroupDown(16 * group_size, 32 * group_size)
        self.down2 = GroupDown(32 * group_size, 64 * group_size)
        self.down3 = GroupDown(64 * group_size, 128 * group_size)
        self.down4 = GroupDown(128 * group_size, 128 * group_size)

    def forward(self, x):
        ...
        return

class GroupModifiedUNet(nn.Module):
    ...
```



注：
1 卷积层通过torch.cat合并
2 BN层通过首尾拼接完成合并
3 其他层视具体情况而定

# 效果
基本能够得到与原本模型相同的结果。

但时间跟原本差不多，why？