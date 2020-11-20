# 单卡多卡并行训练

### torch.nn.DataParallel

一般在使用多GPU的时候，可用`os.environ['CUDA_VISIBLE_DEVICES']`来限制使用的GPU个数，要使用第0和第3编号的GPU，那么只需要在程序中设置：

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
```

如果不想在代码中使用`os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3' `，也可以在运行程序的时候在`python main.py`的前面加上`CUDA_VISIBLE_DEVICES=0,3`。

但是要注意的是，这个参数的设定要保证在模型加载到gpu上之前，一般都是要在程序开始的时候就设定好这个参数。但之后是**如何将模型加载到多GPU上面?** 如下：

```python
# if load model to multi-gpu
model = nn.DataParallel(model)
model = model.cuda()

# or
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
# device_ids用于指定使用的GPU，output_device用于指定汇总梯度的GPU

# loading data
inputs = inputs.cuda()
labels = labels.cuda()
```

**官方示例**：

```python
class DataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        ...

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
```

如果不设定好要使用的`device_ids`的话，程序会**自动找到机器上面可以用的所有的显卡**，然后用于训练，而使用`os.environ['CUDA_VISIBLE_DEVICES']`则**限定了程序可以使用的显卡**。

此外，使用`DataParallel`**每个batch数据被可使用的GPU卡分割**（通常是均分），**也就意味着每个卡上的`batch_size`减小？？**。

**缺点**：在每个训练批次（batch）中，因为模型的权重都是在 一个进程上先算出来然后再把他们分发到每个GPU上，所以网络通信就成为了一个瓶颈，而GPU使用率也通常很低，即为**单进程控制多GPU**。

**<u>此外如何对应上显卡的实际编号和程序里设置的编号，有什么需要注意的地方么</u>**？



### torch.nn.parallel.DistributedDataParallel

pytorch的官网建议使用`DistributedDataParallel`来代替`DataParallel`，是因为`DistributedDataParallel`比`DataParallel`运行的更快, 然后显存分屏的更加均衡。而且`DistributedDataParallel`功能更加强悍。例如分布式的模型(一个模型太大，以至于无法放到一个GPU上运行，需要分开到多个GPU上面执行)。只有`DistributedDataParallel`支持分布式的模型像单机模型那样可以进行多机多卡的运算，详细信息见官方文档 https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html 。

依旧是先设定好`os.environ['CUDA_VISIBLE_DEVICES']`, 然后再进行后面的步骤。因为`DistributedDataParallel`是支持多机多卡的，所以这个需要先初始化一下，代码如下：

```python
# 第一个参数是pytorch支持的通讯后端
# 第二个参数是各个机器之间通讯的方式，单机多卡设置成localhost
# 第三个参数rank标识主机和从机, 如果就一个主机, 设置为0
# 第四个参数world_size是标识使用了几个主机, 一个主机设置为1，否则代码不允许会报错
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
```

如果是单机多卡的情况，可以直接使用以下代码：

```python
'''
初始化:
在启动python脚本后，会通过参数local_rank来声明当前进程使用的是哪个GPU，用于在每个进程中指定不同的device
'''
import torch
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
    
'''
注意, 如果使用torch.distributed.init_process_group这句代码, 直接在pycharm或者别的编辑器中，无法正常运行, 因为这个需要在shell的命令行中运行, 如果想要正确执行这段代码, 假设这段代码的名字是main.py, 可以使用以下的方法进行:
python -m torch.distributed.launch \--nproc_per_node=4 main.py
其中的 --nproc_per_node 参数用于指定为当前主机创建的进程数(设置为所使用的GPU数量即可)，如果是单机多卡，这里node数量为1 ???

注:
这里如果使用了argparse, 一定要在参数里面加上--local_rank, 否则运行还是会出错，之后就和使用DataParallel类似
'''
# --------------------------------------------------------------------------------------------------------

'''
读取数据:
读取数据的时候，要保证一个batch里的数据被均摊到每个进程上，每个进程都能获取到不同的数据，PyTorch已经封装好了这一方法。在初始化 data loader的时候需要使用到torch.utils.data.distributed.DistributedSampler这个特性(在后面多机多卡也会涉及这方面)。
'''
train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
# --------------------------------------------------------------------------------------------------------
'''
模型的初始化:
与nn.DataParallel的初始化方式一样
'''
model = ...
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
# --------------------------------------------------------------------------------------------------------
'''训练'''
optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
# --------------------------------------------------------------------------------------------------------
```

和单进程训练不同的是，**多进程训练需要注意以下事项**：

- 在传数据的时候，一个batch被分到了好几个进程，每个进程在取数据的时候要确保拿到的是不同的数（`DistributedSampler`）
- 要告诉每个进程哪块GPU（`args.local_rank`）
- 在做BatchNormalization的时候要注意同步数据



### DataParallel带来的显存使用不平衡问题

官方给的解决方案是使用 `DistributedDataParallel`来代替 `DataParallel`。

除此之外，还可以使用来自 https://github.com/kimiyoung/transformer-xl 的解决方案`BalancedDataParallel`，其中平衡GPU显存的代码为 https://github.com/Link-Li/Balanced-DataParallel 。原作者是在继承了`DataParallel`类之后进行了改写：

```python
class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)
    ...

'''
similar API as 'DataParallel', 包含三个参数:
第一个参数是第一个GPU要分配多大的batch_size, 注意, 如果使用了梯度累积, 那么这里传入的是每次进行运算的实际batch_size大小 (比如在3个GPU上面跑代码, 但是一个GPU最大只能跑3条数据, 但是因为0号GPU还要做一些数据的整合操作, 于是0号GPU只能跑2条数据, 这样一算, 可以跑的大小是2+3+3=8, 于是可以设置下面的这样的参数：
batch_szie = 8
gpu0_bsz = 2
acc_grad = 1
'''

'''
如果想跑batch size是16, 那就是4+6+6=16了, 这样设置累积梯度为2，如下：
batch_szie = 16
gpu0_bsz = 4
acc_grad = 2
'''

my_net = MyNet()
my_net = BalancedDataParallel(gpu0_bsz // acc_grad, my_net, dim=0).cuda()
```



# 多机多GPU训练

**在单机多gpu可以满足的情况下，不建议使用多机多gpu进行训练，多台机器之间传输数据的时间非常长（取决于网络速度），以致于影响实际训练速度。**

**实战代码（参考）**：

https://github.com/pytorch/examples/blob/master/imagenet/main.py 

https://github.com/edwhere/Distributed-VGG-F



### 初始化

*初始化操作一般在程序刚开始的时候进行*

在进行多机多gpu进行训练的时候, 需要先使用`torch.distributed.init_process_group()`进行初始化. `torch.distributed.init_process_group()`包含四个常用的参数

```python
# backend: 后端, 实际上是多个机器之间交换数据的协议
# init_method: 机器之间交换数据, 需要指定一个主节点, 而这个参数就是指定主节点的
# world_size: 介绍都是说是进程, 实际就是机器的个数, 例如两台机器一起训练的话, world_size就设置为2
# rank: 区分主节点和从节点的, 主节点为0, 剩余的为了1-(N-1), N为要使用的机器的数量, 也就是world_size
```

###### 初始化backend

首先要初始化的是`backend`, 也就是俗称的后端, 在pytorch的官方教程中提供了以下后端（https://pytorch.org/docs/stable/distributed.html#backends）：

| Backend        | `gloo` |      | `mpi` |      | `nccl` |      |
| -------------- | ------ | ---- | ----- | ---- | ------ | ---- |
| Device         | CPU    | GPU  | CPU   | GPU  | CPU    | GPU  |
| send           | ✓      | ✘    | ✓     | ?    | ✘      | ✘    |
| recv           | ✓      | ✘    | ✓     | ?    | ✘      | ✘    |
| broadcast      | ✓      | ✓    | ✓     | ?    | ✘      | ✓    |
| all_reduce     | ✓      | ✓    | ✓     | ?    | ✘      | ✓    |
| reduce         | ✓      | ✘    | ✓     | ?    | ✘      | ✓    |
| all_gather     | ✓      | ✘    | ✓     | ?    | ✘      | ✓    |
| gather         | ✓      | ✘    | ✓     | ?    | ✘      | ✘    |
| scatter        | ✓      | ✘    | ✓     | ?    | ✘      | ✘    |
| reduce_scatter | ✘      | ✘    | ✘     | ✘    | ✘      | ✓    |
| all_to_all     | ✘      | ✘    | ✓     | ?    | ✘      | ✘    |
| barrier        | ✓      | ✘    | ✓     | ?    | ✘      | ✓    |

根据官网的介绍，如果是使用cpu的分布式计算，建议使用`gloo`，因为表中可以看到 `gloo`对cpu的支持是最好的。如果使用gpu进行分布式计算, 建议使用`nccl`。根据官网，并不特别推荐在多gpu的时候使用`mpi`。

对于后端选择好了之后，需要设置一下网络接口，因为多个主机之间一般是使用网络进行交换，会涉及到ip之类，对于`nccl`和`gloo`，一般会自己寻找网络接口，但是特殊情况下（比如网卡过多）需要自己手动设置，在Python中通过以下代码设置：

```python
import os
# 以下二选一, 第一个是使用gloo后端需要设置的, 第二个是使用nccl需要设置的
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
```

自己的网络接口可以通过打开命令行输入`ifconfig`找到对应的ip地址，一般为`em0`, `eth0`, `esp2s0`之类，具体的根据个人情况的填写。



###### 初始化`init_method`

**初始化`init_method`的方法有两种**, 一种是使用TCP进行初始化, 另外一种是使用共享文件系统进行初始化

1. 使用TCP初始化

```python
import torch.distributed as dist

dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=rank, world_size=world_size)
```

注意这里使用格式为`tcp://ip:端口号`，`ip`地址代表主节点的ip地址，也就是`rank`参数为0的主机的ip地址，然后再选择一个空闲的端口号, 这样即可初始化`init_method`。

2. 使用共享文件系统初始化

此方法与硬盘的格式**有关系**，特别是window的硬盘格式和Ubuntu的不一致。

```python
import torch.distributed as dist

dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile', rank=rank, world_size=world_size)
```

根据官网说明，要注意提供的共享文件一开始是不存在的，但是此方法也不会在执行结束后删除文件，所以下次再进行初始化的时候，需要手动删除上次的文件, 操作比较繁琐，推荐先尝试使用TCP初始化。



###### 初始化`rank`和`world_size`

这里需要确保，不同机器的`rank`值不同，主机的`rank`必须为0，而且使用`init_method`的ip一定是`rank`为0的主机；其次`world_size`为实际主机数量，这个数值不能随便设置，如果参与训练的主机数量达不到`world_size`的设置值时，代码不会执行。



###### 初始化过程中注意点

1. 首先是**代码的统一性**，所有的节点上面的代码，建议完全一样，不然有可能会出现问题

2. **初始化的参数强烈建议通过`argparse`模块(命令行参数的形式)输入**，不建议写死在代码中

3. 也**不建议使用pycharm之类的IDE进行代码的运行**，建议使用命令行直接运行

4. 其次是运行代码的命令方面的问题，例如使用下面的命令在主节点上（设置`rank`为0，同时设置了使用两个主机）运行代码`distributed.py`：

```text
python distributed.py -bk nccl -im tcp://10.10.10.1:12345 -rn 0 -ws 2
```

当在节点运行的时候，代码如下:

```text
python distributed.py -bk nccl -im tcp://10.10.10.1:12345 -rn 1 -ws 2
```

一定要注意的是, **只能修改`rank`的值，其他的值一律不得修改**，否则程序会卡死。



### 数据的处理 - DataLoader

多机多卡中数据的处理和正常的代码的数据处理非常类似，但因为多机多卡涉及到效率问题，所以这里一般使用`torch.utils.data.distributed.DistributedSampler`来规避数据传输的问题，见如下代码:

```python
print("Initialize Dataloaders...")
# Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Initialize Datasets. STL10 will automatically download if not present
trainset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
valset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

# Create DistributedSampler to handle distributing the dataset across nodes when training
# This can only be called after torch.distributed.init_process_group is called
# 这一句就是和平时使用有点不一样的地方
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# Create the Dataloaders to feed data to the training and validation steps
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=False, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
```

这段代码和平时写的很类似，唯一的区别就是这里先将`trainset`送到了`DistributedSampler`中创造了一个`train_sampler`，然后在构造`train_loader`的时候，参数中传入了一个`sampler=train_sampler`。意图是让不同节点的机器加载自己本地的数据进行训练，也就是说进行多机多卡训练的时候，不再是从主节点分发数据到各个从节点，而是各个从节点自己从自己的硬盘上读取数据，**可看作为Distributed Learning 或者 Federated Learning**。

如果直接让各个节点自己读取自己的数据，特别是在训练的时候经常是要打乱数据集进行训练，这样就会导致不同的节点加载的数据混乱，所以使用`DistributedSampler`来创造一个`sampler`提供给`DataLoader`，`sampler`的作用是自定义一个数据的编号，然后让`DataLoader`按照这个编号来提取数据放入到模型中训练，其中`sampler`参数和`shuffle`参数不能同时指定，如果这个时候还想要可以随机的输入数据，可以在`DistributedSampler`中指定`shuffle`参数，具体的可以参考官网最后的`DistributedSampler`介绍。



### 模型的处理

模型的处理与单机多卡没有太大区别，还是下面的代码，但是**注意要提前想把模型加载到gpu**，然后才可以加载到`DistributedDataParallel`

```text
model = model.cuda()
model = nn.parallel.DistributedDataParallel(model)
```



### 模型的保存与加载

这里引用了pytorch官方教程 https://pytorch.org/tutorials/intermediate/ddp_tutorial.html 的一段代码：

```python
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```

以上代码核心为`dist.barrier()`，来自`torch.distributed.barrier()`，根据pytorch的官网的介绍，此函数的功能是同步所有的进程，直到整组(也就是所有节点的所有GPU)到达这个函数的时候，才会执行后面的代码。此外，可以看到，在保存模型的时候是只找`rank`为0的点保存模型(也可以保存其他节点的模型)，然后在加载模型的时候，首先同步所有节点，然后为所有的节点加载上模型，之后在进行下一步的时候，还需要同步，以保证所有的节点都读完了模型。



# 总结与讨论

### 多卡与单卡效率问题

双卡效率基本低于单卡两倍，比较好的版本可能达到3.2x+/4卡的提速。可能的变量有：

1. 并行方式为数据并行，同时统一`batch_size`不变意味着单卡的`batch_size`变小，GPU利用率可能会下降
2. 多个GPU之间是需要通信/计算/回传梯度的，此时不同硬件、不同交互方式(蝶式\环式)的IO开销是不同的。
3. 多GPU可能涉及某些操作(sysc BN)更改，效率也会影响



### <u>跨卡同步 Batch Normalization</u>

为什么要同步BN? 详情见https://zhuanlan.zhihu.com/p/40496177

现有的标准 Batch Normalization 因为使用数据并行（Data Parallel），是单卡的实现模式，只对单个卡上对样本进行归一化，相当于减小了批量大小（batch-size）（详见BN工作原理部分）。 对于比较消耗显存的训练任务时，往往单卡上的相对批量过小，影响模型的收敛效果。 在图像语义分割的实验中，可能出现使用大模型的效果反而变差，实际上就是BN在作怪。 跨卡同步 Batch Normalization 可以使用全局的样本进行归一化，这样相当于’增大‘了批量大小，这样训练效果不再受到使用 GPU 数量的影响。 最近在图像分割、物体检测的论文中，使用跨卡BN也会显著地提高实验效果。

###### 数据并行 DataParallel

深度学习平台在多卡（GPU）运算的时候都是采用的数据并行（DataParallel），如下图:

![2019-08-17-2](https://niecongchong.github.io/img/2019-08-17-2.jpg)

每次迭代，输入被等分成多份，然后分别在不同的卡上前向（forward）和后向（backward）运算，并且求出梯度，在迭代完成后合并 梯度、更新参数，再进行下一次迭代。因为在前向和后向运算的时候，每个卡上的模型是单独运算的，所以相应的Batch Normalization 也是在卡内完成，所以实际BN所归一化的样本数量仅仅局限于卡内，**相当于批量大小（batch-size）减小了**。

###### 跨卡同步 (Cross-GPU Synchronized) 或同步BN (SyncBN)

跨卡同步BN的关键是在前向运算的时候拿到全局的均值和方差，在后向运算时候得到相应的全局梯度。 最简单的实现方法是先同步求均值，再发回各卡然后同步求方差，但是这样就同步了两次。实际上只需要同步一次就可以，因为总体`batch_size`对应的均值和方差可以通过加权每张GPU中均值和方差得到（如下图）。在反向传播时也一样需要同步一次梯度信息，详情见 https://arxiv.org/pdf/1803.08904.pdf 。

![2019-08-17-3](https://niecongchong.github.io/img/2019-08-17-3.jpg)

这样在前向运算的时候，只需要先在各卡上算出，再跨卡求出全局的和即可得到正确的均值和方差， 同理我们在后向运算的时候只需同步一次，求出相应的梯度。

###### SyncBN现有资源

- [jianlong-yuan/syncbn-tensorflow](https://link.zhihu.com/?target=https%3A//github.com/jianlong-yuan/syncbn-tensorflow)重写了TensorFlow的官方方法，可以做个实验验证一下。
- [旷视科技：CVPR 2018 旷视科技物体检测冠军论文——大型Mini-Batch检测器MegDet](https://zhuanlan.zhihu.com/p/37847559)
- [tensorpack/tensorpack](https://link.zhihu.com/?target=https%3A//github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)里面有该论文的代码实现。

###### 为什么不用全局mean和var（整个数据集的），之后设置好网络中所有mean和var且不更新？

因为训练过程中weights会变，这个时候mean和var就会变。如果用global states的话就相当于没有batchnorm这样的话就只能用小learning rate。最后的效果不会有batchnorm的好。

###### 为什么不进行多卡同步？

现有框架BatchNorm的实现都是只考虑了single gpu。也就是说BN使用的均值和标准差是单个gpu算的，相当于缩小了mini-batch size。这是因为，1）因为没有sync的需求，因为对于大多数视觉问题，单gpu上的mini-batch已经够大了，完全不会影响结果。2）影响训练速度，BN layer通常是在网络结构里面广泛使用的，这样每次都同步一下GPUs，十分影响训练速度。

# Reference

[知乎专栏 - pytorch多gpu并行训练](https://zhuanlan.zhihu.com/p/86441879)

[Nick blog - Multi-GPU下的Batch normalize跨卡同步](https://niecongchong.github.io/2019/08/17/Multi-GPU%E4%B8%8B%E7%9A%84Batch-normalize%E8%B7%A8%E5%8D%A1%E5%90%8C%E6%AD%A5/)

[PyTorch - DATA PARALLELISM](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

[PyTorch - torch.nn](https://pytorch.org/docs/stable/nn.html#distributeddataparallel)

[PyTorch - DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization)

[PyTorch - ImageNet examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

[PyTorch - Official Tutorials](https://pytorch.org/tutorials/)

[Distributed VGG-F](https://github.com/edwhere/Distributed-VGG-F)

[PyTorch - GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

