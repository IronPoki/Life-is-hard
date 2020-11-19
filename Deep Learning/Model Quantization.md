# Why

Storage, memory, power and latency requirements of a deep neural network is limited, especially on mobile or embedded devices.

# What

Convert model parameters, intermediate variables from continuous space to discrete (int 8/16). Basically, it uses less bits to approximate 32 bit data to reduce model size, memory requirements, accelerate model inference and maintain model accuracy at the mean time.

![image-20201119135255285](C:\Users\Cheng Zhu\AppData\Roaming\Typora\typora-user-images\image-20201119135255285.png)



### Pros

- **减小模型尺寸**，如8位整型量化可减少75%的模型大小，32 --> 8

- **减少存储空间**，在边缘侧存储空间不足时更具有意义

- **易于在线升级**，模型更小意味着更加容易传输

- **减少内存耗用**，更小的模型大小意味着不需要更多的内存

- **加快推理速度**，访问一次32位浮点型可以访问四次int8整型，整型运算比浮点型运算更快

- **减少设备功耗**，内存耗用少了推理速度快了自然减少了设备功耗

- **支持微处理器**，有些微处理器属于8位的，低功耗运行浮点运算速度慢，需要进行8bit量化

  

### Cons

- **模型量化增加了操作复杂度**，在量化时需要做一些特殊的处理，否则精度损失更严重
- **模型量化会损失一定的精度**，虽然在微调后可以减少精度损失，但推理精度确实下降。需要对模型进行**量化重训**。



# How

模型量化桥接了定点与浮点，建立了一种有效的数据映射关系，使得以较小的精度损失代价获得了较好的收益，要弄懂模型量化的原理就是要弄懂这种数据映射关系。

### 浮点数，定点数

**定点数和浮点数的定义** https://www.jianshu.com/p/43830cdcca30， https://zhuanlan.zhihu.com/p/63897066：一种数约定数值的小数点固定在某一位置，称为定点表示法，简称为**定点数**，int本质是小数点位于末尾的32位定点数。对应的另一种方法，小数点可以任意浮动，称为浮点表示法，简称为**浮点数**。

For example: 
$$
浮点数转定点数:int8(10)=float32(1.231)\times2^{(3)}
$$

$$
定点数转浮点数:float32(1.250)=int8(10)\div{2^{(3)}}
$$

同样的int8数，会因为量化系数的不同代表不同的float32值。
$$
定点数: Q = \frac{R}{S} + Z
$$

$$
浮点数: R = (Q - Z) * S
$$

R表示真实的浮点值，Q表示量化后的定点值（int8即为 [-128, 127]），Z表示0浮点值对应的量化定点值，S则为定点量化后可表示的最小刻度，S和Z的求值公式如下：
$$
S: S=\frac{R_{max} - R_{min}}{Q_{max}-Q_{min}}
\qquad
Z: Z=Q_{max} - R_{max}\div{S}
$$

$$
R_{max}:最大的浮点值  \qquad R_{min}:最小的浮点值 \qquad Q_{max}:最大的定点值  \qquad Q_{min}:最小的定点值  \qquad
$$

这里的S和Z均是量化参数，而Q和R均可由公式进行求值，不管是量化后的Q还是反推求得的浮点值R，如果它们超出各自可表示的最大范围，那么均需要进行截断处理。而浮点值0在神经网络里有着举足轻重的意义，比如padding就是用的0，因而必须有精确的整型值来对应浮点值0。

### 模型量化

### tensorflow
#### Post-training quantization

tensorflow训练后量化是针对已训练好的模型来说的，针对大部分我们已训练未做任何处理的模型来说均可用此方法进行模型量化，而tensorflow提供了一整套完整的模型量化工具，如**T**ensorFlow Lite **O**ptimizing **CO**nverter（**toco**命令工具）以及TensorFlow Lite converter（API源码调用接口）。

| Technique              | Benefits                               | Hardware            |
| ---------------------- | -------------------------------------- | ------------------- |
| Post training 'hybrid' | 4x smaller, 2-3x speed up, accuracy    | CPU                 |
| Post training integer  | 4x smaller, More speedup               | CPU, Edge TPU, etc. |
| Post training fp16     | 2x smaller, Potential FPU acceleration | CPU/GPU             |

###### 1. 混合量化 - 仅量化权重

该方式将浮点型的权重量化为int8整型，可将模型大小直接减少75%、提升推理速度最大3倍。该方式在推理的过程中，需要将int8量化值反量化为浮点型后再进行计算，如果某些Ops不支持int8整型量化，那么其保存的权重依然是浮点型的，即部分支持int8量化的Ops其权重保存为int8整型且存在quantize和dequantize操作，否则依然是浮点型的，因而称该方式为混合量化。该方式可达到近乎全整型量化的效果，但存在quantize和dequantize操作其速度依然不够理想。方法如下：

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

###### 2. 全整型量化 - 权重和激活值都进行量化

该方式则试图将权重、激活值及输入值均全部做int8量化，并且将所有模型运算操作置于int8下进行执行，以达到最好的量化效果。为了达到此目的，我们需要一个具有代表性的小数据集，用于统计激活值和输入值等的浮点型范围，以便进行精准量化，方法如下：

```python
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

全整型量化的输入输出依然是浮点型的，但如果某些Ops未实现该方法，则转化是没问题的且其依然会自动保存为浮点型，这就要求我们的硬件支持这样的操作，为了防止这样的问题出现，我们可以在代码里添加如下语句强制检查并在Ops不支持全整型量化时进行报错。

```python
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
```

###### 3. 半精度float16量化 - 仅量化权重

该方式是将权重量化为半精度float16形式，其可以减少一半的模型大小、相比于int8更小的精度损失，如果硬件支持float16计算的话那么其效果更佳，这种方式是google近段时间提供的，其实现方式也比较简单，仅需在代码中调用如下接口即可：

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
tflite_quant_model = converter.convert()
```

当在CPU运行时，半精度量化也需要像int8量化一样进行反量化到float32再进行计算，但在GPU则不需要，因为GPU可以支持float16运算，但官方声明float16量化没有int8量化性价比高。

#### Quantization-aware training

tensorflow量化感知训练是一种伪量化的过程，它是在可识别的某些操作内嵌入伪量化节点（fake quantization nodes），用以统计训练时流经该节点数据的最大最小值，便于在使用TOCO转换tflite格式时量化使用并减少精度损失，其参与模型训练的前向推理过程令模型获得量化损失，但梯度更新需要在浮点下进行因而其并不参与反向传播过程。某些操作无法添加伪量化节点，这时候就需要人为的去统计某些操作的最大最小值，但如果统计不准那么将会带来较大的精度损失，因而需要较谨慎检查哪些操作无法添加伪量化节点。值得注意的是，伪量化节点的意义在于统计流经数据的最大最小值并参与前向传播提升精确度，但其在TOCO工具转换为量化模型后，其工作原理还是与训练后量化方式一致的。

官方文档：https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html

中文文档：https://zhuanlan.zhihu.com/p/144870688

#### Post-training quantization vs Quantization-aware training

tensorflow训练后量化和量化感知训练是两种不同的量化方式，前者是一种offline的方式，而后者则是一种online的方式，由于深度学习神经网络DNN对噪声和扰动比较鲁棒且训练出来的模型权重往往落入一个有限的区间范围内，因而这两者均可达到以较少精度损失达到模型量化的目的，各自的优缺点如下：

- 两者均可达到模型量化的作用
- 两者的推理工作原理是一样的
- 两者都可工作在Tensorflow lite推理框架下并进行相应加速
- 训练后量化工作量稍微简单些，而量化感知训练工作量更繁琐一些
- 量化感知训练比训练后量化损失的精度更少，官方推荐使用量化感知训练方式



### Pytorch

官方文档：https://pytorch.org/docs/stable/quantization.html  **注：目前PyTorch的量化工具仅支持1.3及以上版本**。



###### 数据类型：

- weight的8 bit量化 ：data_type = qint8，数据范围为[-128, 127]
- activation的8 bit量化：data_type = quint8，数据范围为[0, 255]

bias一般是不进行量化操作的，仍然保持float32的数据类型，还有一个需要提前说明的，weight在浮点模型训练收敛之后一般就已经固定住了，所以根据原始数据就可以直接量化，然而**activation会因为每次输入数据的不同，导致数据范围每次都是不同的**，所以针对这个问题，在量化过程中专门会有一个校准过程，即提前准备一个小的**校准数据集**（类似生成.wk使用的输入图片），在测试这个校准数据集的时候会记录每一次的activation的数据范围，然后根据记录值确定一个固定的范围。



###### 支持后端：

- 具有 AVX2 支持或更高版本的 x86 CPU：fbgemm
- ARM CPU：qnnpack

通过如下方式进行设置：

```python
q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend
qconfig = torch.quantization.get_default_qconfig(q_backend)   
```

打印输出可得：

```python
QConfig(activation=functools.partial(<class 'torch.quantization.observer.HistogramObserver'>, reduce_range=False), 
            weight=functools.partial(<class 'torch.quantization.observer.MinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
```

可以看出qnnpack 的量化方式：activation量化的数据范围是通过记录每次运行数据的最大最小值的统计直方图导出的，weight为per-layer的对称量化，整型数据范围为[-128,127]，直接用最后收敛的浮点数据的最大最小值作为范围进行量化，其他信息仅从这个打印信息暂时还得不到。



###### 量化方法

1. **Post Training Dynamic Quantization**：这是最简单的一种量化方法，Post Training指的是在浮点模型训练收敛之后进行量化操作，其中weight被提前量化，而activation在前向推理过程中被动态量化，即每次都要根据实际运算的浮点数据范围每一层计算一次scale和zero_point，然后进行量化；
2. **Post Training Static Quantization**：第一种不是很常见，一般说的Post Training Quantization指的其实是这种静态的方法，而且这种方法是最常用的，其中weight跟上述一样也是被提前量化好的，然后activation也会基于之前校准过程中记录下的固定的scale和zero_point进行量化，整个过程不存在量化参数*(*scale和zero_point)的再计算；
3. **Quantization Aware Training**：对于一些模型在浮点训练+量化过程中精度损失比较严重的情况，就需要进行量化感知训练，即在训练过程中模拟量化过程，数据虽然都是表示为float32，但实际的值的间隔却会受到量化参数的限制。

至于为什么不在一开始训练的时候就模拟量化操作是因为8bit精度不够容易导致模型无法收敛，甚至直接使用16bit进行from scrach的量化训练都极其容易导致无法收敛。**<u>有一些trick可以解决这个问题么？</u>**



###### 量化流程

以最常用的Post Training (Static) Quantization为例：

1. **准备模型：**准备一个训练收敛了的浮点模型**，**用**QuantStub**和**DeQuantstub**模块指定需要进行量化的位置；
2. **模块融合：**将一些相邻模块进行融合以提高计算效率，比如conv+relu或者conv+batch normalization+relu，最常提到的BN融合指的是conv+bn通过计算公式将bn的参数融入到weight中，并生成一个bias；

3. **确定量化方案：**这一步需要指定量化的后端(qnnpack/fbgemm/None)，量化的方法(per-layer/per-channel，对称/非对称)，activation校准的策略(最大最小/移动平均)；

4. **activation校准：**利用torch.quantization.prepare() 插入将在校准期间观察激活张量的模块，然后将校准数据集灌入模型，利用校准策略得到每层activation的scale和zero_point并存储；

5. **模型转换：**使用 torch.quantization.convert(）函数对整个模型进行量化的转换。 这其中包括：它量化权重，计算并存储要在每个激活张量中使用的scale和zero_point，替换关键运算符的量化实现；

   

###### 量化工具

1. **torch.quantization**：最基础的量化库，里面包含模型直接转换函数torch.quantization.quantize，量化训练函数torch.quantization.quantize_qat，准备校准函数torch.quantization.prepare等一系列工具
2. **quantize_per_tensor**：per-ayer量化，需要手动指定scale, zero_point和数据类型dtype；
3. **quantize_per_channel**：per-channel量化，除了需要指定上述三个参数之外，还需要额外指定执行per-channel量化的维度；
4. **torch.nn.intrinsic.quantized：**提供了很多已经融合好的模块，如ConvBn2d，ConvBnReLU2d，直接对这些模型进行量化
5. 其余的如**torch.nn.quantized，torch.nn.quantized.functional**......



###### Quantization-Aware Training 相关模块

1. **torch.nn.qat：**支持全连接和二维卷积操作，可以实现在float32数据中模仿量化的操作，即量化+反量化；
2. **torch.nn.intrinsic.qat：**对融合好的层进行量化训练。



### Summary

可以继续进行的研究：

- 8bit的from scracth训练，这样就可以实现训练加速，前段时间研读了商汤的一篇相关论文，很有意思；
- full-quantized method，去掉中间大量的quantize和de-quantize操作，bias也进行量化，这样不可避免会丧失一定的灵活性(研究一下add这块怎么对scale进行对齐就能感受到)，但整个过程中没有float32参与了，硬件运行是更高效的，尤其对于FPGA这种对float不是很友好的硬件；
- 将乘除法用移位来代替，在保证精度的前提下进一步提升硬件运行效率；这个论文目前比较多，之前在ICCV2019的poster现场看到不少；
- 无需提供数据或只需极少量数据就能对每一层的activation进行校准，这一步对精度影响还是蛮大的，但是对于一些比较隐私的数据(如医学)，提供数据还是比较难的，所以如果能够从模型本身获取得到activation相关的信息然后进行校准是很有意义的 [Ref: https://arxiv.org/abs/2001.00281]
- 模型量化向其他任务上的迁移，如目标检测，语义分割，人体位姿检测，NLP任务，如何在不进行量化训练的前提下保持较高精度(可以实现快速部署)；



# Reference

https://zhuanlan.zhihu.com/p/79744430

