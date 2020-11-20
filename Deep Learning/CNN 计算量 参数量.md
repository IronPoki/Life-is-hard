这里暂时先忽略bias，但实际中必须要考虑

# 卷积层的参数量

卷积层的参数主要来自filter/kernel，计算公式为:
$$
param\_conv = filter\_size \times input\_channels \times filter\_num
$$


# 全连接层参数量

全连接层参数冗余，参数量可以占整个网络参数80%左右，具体计算公式类似卷积层：

假设：

input: 7 x 7 x 512

output: 1 x 4096

则是使用4096个 (7 x 7 x 512) 的filter去做卷积 (可以理解为一个卷积层)。
$$
params\_dense = 7 \times 7\times512\times4,096 = 102,760,448
$$


# 卷积层的计算量

一个batch向前传播计算量就有：
$$
computation = output\_size \times filter_size \times batch\_num
$$


# 全连接层计算量

**与参数量一致**。



# Summary

1. 卷积层的参数量与计算量的差异体现出权值共享的必要性；
2. 需要减少网络参数量时（模型大小）主要是针对全连接层；
3. 需要减少网络计算量时（模型加速，计算优化）主要是针对卷积层。