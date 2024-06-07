import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn import init  # 导入PyTorch的初始化模块
import functools  # 导入functools模块，提供了一些有用的函数，可以用来改变或扩展原有函数的一些功能
from torch.optim import lr_scheduler  # 导入PyTorch的学习率调度器模块
from models.layers.mesh_conv import MeshConv  # 导入自定义的MeshConv模块，用于在网格上进行卷积操作
import torch.nn.functional as F  # 导入PyTorch的函数接口模块
from models.layers.mesh_pool import MeshPool  # 导入自定义的MeshPool模块，用于在网格上进行池化操作
from models.layers.mesh_unpool import MeshUnpool  # 导入自定义的MeshUnpool模块，用于在网格上进行反池化操作


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance', num_groups=1):
    # 定义一个函数，用于获取指定类型的标准化层
    if norm_type == 'batch':
        # 如果标准化类型为'batch'，则使用批量标准化层，affine参数为True表示在标准化后添加偏置和缩放因子
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        # 如果标准化类型为'instance'，则使用实例标准化层，affine参数为False表示在标准化后不添加偏置和缩放因子
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        # 如果标准化类型为'group'，则使用组标准化层，affine参数为True表示在标准化后添加偏置和缩放因子，num_groups参数表示分组数量
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        # 如果标准化类型为'none'，则使用NoNorm层，即不进行任何标准化操作
        norm_layer = NoNorm
    else:
        # 如果标准化类型不在已知的类型中，抛出未实现错误
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer  # 返回获取到的标准化层

def get_norm_args(norm_layer, nfeats_list):
    # 定义一个函数，用于获取标准化层的参数
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        # 如果标准化层的名称为'NoNorm'，则创建一个假的参数列表，列表长度与特征列表长度相同
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        # 如果标准化层的名称为'GroupNorm'，则创建一个参数列表，其中每个参数的'num_channels'值对应特征列表中的一个特征
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        # 如果标准化层的名称为'BatchNorm'，则创建一个参数列表，其中每个参数的'num_features'值对应特征列表中的一个特征
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        # 如果标准化层的名称不在已知的名称中，抛出未实现错误
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args  # 返回获取到的参数列表


class NoNorm(nn.Module): 
    # 定义一个NoNorm类，继承自PyTorch的Module类，表示不进行任何标准化操作的层
    def __init__(self, fake=True):
        # 构造函数，接受一个可选参数fake，默认为True
        self.fake = fake  # 将fake参数保存为实例属性
        super(NoNorm, self).__init__()  # 调用父类的构造函数
    def forward(self, x):
        # 定义前向传播函数，接受一个输入x，直接返回x，即不进行任何处理
        return x
    def __call__(self, x):
        # 定义__call__方法，使得实例可以像函数一样被调用，接受一个输入x，调用forward方法处理并返回结果
        return self.forward(x)

def get_scheduler(optimizer, opt):
    # 定义一个函数，用于获取指定类型的学习率调度器
    if opt.lr_policy == 'lambda':
        # 如果学习率策略为'lambda'，则使用LambdaLR调度器
        def lambda_rule(epoch):
            # 定义一个函数，用于计算每个epoch的学习率
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # 创建LambdaLR调度器，lr_lambda参数为上面定义的函数
    elif opt.lr_policy == 'step':
        # 如果学习率策略为'step'，则使用StepLR调度器，step_size参数为学习率衰减的步长，gamma参数为学习率衰减的因子
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        # 如果学习率策略为'plateau'，则使用ReduceLROnPlateau调度器，mode参数为'min'表示当某指标停止下降时，学习率将减少，factor参数为学习率减少的因子，threshold参数为阈值，patience参数为耐心值，即当学习停滞不前时，等待的epoch数量
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        # 如果学习率策略不在已知的策略中，抛出未实现错误
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler  # 返回获取到的调度器


def init_weights(net, init_type, init_gain):
    # 定义一个函数，用于初始化网络的权重
    def init_func(m):
        # 定义一个函数，用于初始化模块的权重
        classname = m.__class__.__name__  # 获取模块的类名
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 如果模块有权重属性，并且是卷积层或线性层
            if init_type == 'normal':
                # 如果初始化类型为'normal'，则使用正态分布初始化权重
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                # 如果初始化类型为'xavier'，则使用xavier正态分布初始化权重
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                # 如果初始化类型为'kaiming'，则使用kaiming正态分布初始化权重
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                # 如果初始化类型为'orthogonal'，则使用正交初始化权重
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                # 如果初始化类型不在已知的类型中，抛出未实现错误
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            # 如果模块是BatchNorm2d层，使用正态分布初始化权重，使用常数初始化偏置
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # 对网络的每个模块应用初始化函数


def init_net(net, init_type, init_gain, gpu_ids):
    # 定义一个函数，用于初始化网络
    if len(gpu_ids) > 0:
        # 如果gpu_ids列表的长度大于0，表示有可用的GPU
        assert(torch.cuda.is_available())  # 断言CUDA是否可用
        net.cuda(gpu_ids[0])  # 将网络移动到第一个GPU上
        net = net.cuda()  # 将网络移动到CUDA设备上
        net = torch.nn.DataParallel(net, gpu_ids)  # 使用DataParallel包装网络，使其可以在多个GPU上并行运行
    if init_type != 'none':
        # 如果初始化类型不为'none'，则初始化网络的权重
        init_weights(net, init_type, init_gain)
    return net  # 返回初始化后的网络


def define_classifier(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    # 定义一个函数，用于创建分类器
    net = None  # 初始化网络为None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)  # 获取标准化层

    if arch == 'mconvnet':
        # 如果架构为'mconvnet'，则创建MeshConvNet网络
        net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    elif arch == 'meshunet':
        # 如果架构为'meshunet'，则创建MeshEncoderDecoder网络
        down_convs = [input_nc] + ncf  # 定义下采样卷积层的通道数
        up_convs = ncf[::-1] + [nclasses]  # 定义上采样卷积层的通道数
        pool_res = [ninput_edges] + opt.pool_res  # 定义池化层的分辨率
        net = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                                 transfer_data=True)
    else:
        # 如果架构不在已知的架构中，抛出未实现错误
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)  # 初始化网络并返回

def define_loss(opt):
    # 定义一个函数，用于创建损失函数
    if opt.dataset_mode == 'classification':
        # 如果数据集模式为'classification'，则使用交叉熵损失
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        # 如果数据集模式为'segmentation'，则使用交叉熵损失，并忽略标签为-1的数据
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    return loss  # 返回创建的损失函数

##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################
class MeshConvNet(nn.Module):
    """用于学习全局形状描述符（分类）的网络
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()  # 调用父类的初始化函数
        self.k = [nf0] + conv_res  # 定义卷积层的通道数
        self.res = [input_res] + pool_res  # 定义池化层的分辨率
        norm_args = get_norm_args(norm_layer, self.k[1:])  # 获取标准化层的参数

        for i, ki in enumerate(self.k[:-1]):
            # 对于每个卷积层，创建一个MResConv层，一个标准化层和一个MeshPool层
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])  # 创建一个平均池化层
        # self.gp = torch.nn.MaxPool1d(self.res[-1])  # 创建一个最大池化层
        self.fc1 = nn.Linear(self.k[-1], fc_n)  # 创建一个全连接层
        self.fc2 = nn.Linear(fc_n, nclasses)  # 创建一个全连接层

    def forward(self, x, mesh):
        # 定义前向传播函数
        for i in range(len(self.k) - 1):
            # 对于每个卷积层，进行卷积、标准化、激活和池化操作
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)  # 对最后的输出进行池化操作
        x = x.view(-1, self.k[-1])  # 将输出的形状变为(batch_size, self.k[-1])

        x = F.relu(self.fc1(x))  # 对输出进行全连接和激活操作
        x = self.fc2(x)  # 对输出进行全连接操作
        return x  # 返回输出

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()  # 调用父类的初始化函数
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.skips = skips  # 跳跃连接的数量
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)  # 创建一个MeshConv层
        for i in range(self.skips):
            # 对于每个跳跃连接，创建一个BatchNorm2d层和一个MeshConv层
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        # 定义前向传播函数
        x = self.conv0(x, mesh)  # 对输入进行卷积操作
        x1 = x  # 保存卷积后的结果
        for i in range(self.skips):
            # 对于每个跳跃连接，进行标准化、激活和卷积操作
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1  # 将卷积后的结果和原始结果相加
        x = F.relu(x)  # 对结果进行激活操作
        return x  # 返回结果


class MeshEncoderDecoder(nn.Module):
    """用于全卷积任务（分割）的网络
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()  # 调用父类的初始化函数
        self.transfer_data = transfer_data  # 是否在解码器中传递数据
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)  # 创建一个MeshEncoder对象
        unrolls = pools[:-1].copy()  # 复制池化层的分辨率列表，并去掉最后一个元素
        unrolls.reverse()  # 反转列表
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)  # 创建一个MeshDecoder对象

    def forward(self, x, meshes):
        # 定义前向传播函数
        fe, before_pool = self.encoder((x, meshes))  # 对输入进行编码操作，并获取编码后的特征和池化前的特征
        fe = self.decoder((fe, meshes), before_pool)  # 对编码后的特征进行解码操作
        return fe  # 返回解码后的特征

    def __call__(self, x, meshes):
        # 定义__call__方法，使得对象可以像函数一样被调用
        return self.forward(x, meshes)  # 调用前向传播函数

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0):
        super(DownConv, self).__init__()  # 调用父类的初始化函数
        self.bn = []  # 初始化标准化层列表
        self.pool = None  # 初始化池化层
        self.conv1 = MeshConv(in_channels, out_channels)  # 创建一个MeshConv层
        self.conv2 = []  # 初始化卷积层列表
        for _ in range(blocks):
            # 对于每个块，创建一个MeshConv层，并添加到卷积层列表中
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            # 对于每个块，创建一个InstanceNorm2d层，并添加到标准化层列表中
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)  # 如果需要池化，则创建一个MeshPool层

    def __call__(self, x):
        # 定义__call__方法，使得对象可以像函数一样被调用
        return self.forward(x)  # 调用前向传播函数

    def forward(self, x):
        # 定义前向传播函数
        fe, meshes = x  # 获取输入的特征和网格
        x1 = self.conv1(fe, meshes)  # 对特征进行卷积操作
        if self.bn:
            x1 = self.bn[0](x1)  # 如果有标准化层，则对卷积后的结果进行标准化操作
        x1 = F.relu(x1)  # 对标准化后的结果进行激活操作
        x2 = x1  # 保存激活后的结果
        for idx, conv in enumerate(self.conv2):
            # 对于每个卷积层，进行卷积、标准化、加法和激活操作
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)  # 将输出的形状变为(batch_size, out_channels, num_points)
        before_pool = None
        if self.pool:
            # 如果有池化层，则保存池化前的结果，并对输出进行池化操作
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool  # 返回输出和池化前的结果


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()  # 调用父类的初始化函数
        self.residual = residual  # 是否使用残差连接
        self.bn = []  # 初始化标准化层列表
        self.unroll = None  # 初始化反池化层
        self.transfer_data = transfer_data  # 是否在卷积层中传递数据
        self.up_conv = MeshConv(in_channels, out_channels)  # 创建一个MeshConv层
        if transfer_data:
            # 如果需要传递数据，则创建一个输入通道数为2 * out_channels的MeshConv层
            self.conv1 = MeshConv(2 * out_channels, out_channels)
        else:
            # 否则，创建一个输入通道数为out_channels的MeshConv层
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []  # 初始化卷积层列表
        for _ in range(blocks):
            # 对于每个块，创建一个MeshConv层，并添加到卷积层列表中
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            # 如果需要进行批量标准化，则创建相应数量的InstanceNorm2d层，并添加到标准化层列表中
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)  # 如果需要进行反池化，则创建一个MeshUnpool层

    def __call__(self, x, from_down=None):
        # 定义__call__方法，使得对象可以像函数一样被调用
        return self.forward(x, from_down)  # 调用前向传播函数

    def forward(self, x, from_down):
        # 定义前向传播函数
        from_up, meshes = x  # 获取输入的特征和网格
        x1 = self.up_conv(from_up, meshes).squeeze(3)  # 对特征进行卷积操作，并将输出的形状变为(batch_size, out_channels, num_points)
        if self.unroll:
            x1 = self.unroll(x1, meshes)  # 如果有反池化层，则对输出进行反池化操作
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)  # 如果需要传递数据，则将输出和下层的输出进行拼接
        x1 = self.conv1(x1, meshes)  # 对拼接后的结果进行卷积操作
        if self.bn:
            x1 = self.bn[0](x1)  # 如果有标准化层，则对卷积后的结果进行标准化操作
        x1 = F.relu(x1)  # 对标准化后的结果进行激活操作
        x2 = x1  # 保存激活后的结果
        for idx, conv in enumerate(self.conv2):
            # 对于每个卷积层，进行卷积、标准化、加法和激活操作
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)  # 将输出的形状变为(batch_size, out_channels, num_points)
        return x2  # 返回输出

class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()  # 调用父类的初始化函数
        self.fcs = None  # 初始化全连接层
        self.convs = []  # 初始化卷积层列表
        for i in range(len(convs) - 1):
            # 对于每个卷积层，创建一个DownConv层，并添加到卷积层列表中
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
        self.global_pool = None  # 初始化全局池化层
        if fcs is not None:
            # 如果有全连接层，则创建相应数量的全连接层和标准化层，并添加到相应的列表中
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)  # 将卷积层列表转换为模块列表
        reset_params(self)  # 重置模型参数

    def forward(self, x):
        # 定义前向传播函数
        fe, meshes = x  # 获取输入的特征和网格
        encoder_outs = []  # 初始化编码器输出列表
        for conv in self.convs:
            # 对于每个卷积层，进行卷积操作，并将卷积前的结果添加到编码器输出列表中
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            # 如果有全连接层，则进行全局池化操作，并将结果转换为一维向量
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                # 对于每个全连接层，进行全连接操作和标准化操作
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)  # 对于非最后一层全连接层，进行激活操作
        return fe, encoder_outs  # 返回最后的特征和编码器输出列表

    def __call__(self, x):
        # 定义__call__方法，使得对象可以像函数一样被调用
        return self.forward(x)  # 调用前向传播函数


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()  # 调用父类的初始化函数
        self.up_convs = []  # 初始化卷积层列表
        for i in range(len(convs) - 2):
            # 对于每个卷积层，创建一个UpConv层，并添加到卷积层列表中
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)  # 创建最后的卷积层
        self.up_convs = nn.ModuleList(self.up_convs)  # 将卷积层列表转换为模块列表
        reset_params(self)  # 重置模型参数

    def forward(self, x, encoder_outs=None):
        # 定义前向传播函数
        fe, meshes = x  # 获取输入的特征和网格
        for i, up_conv in enumerate(self.up_convs):
            # 对于每个卷积层，进行卷积操作，并将卷积前的结果添加到编码器输出列表中
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))  # 对最后的特征进行卷积操作
        return fe  # 返回最后的特征

    def __call__(self, x, encoder_outs=None):
        # 定义__call__方法，使得对象可以像函数一样被调用
        return self.forward(x, encoder_outs)  # 调用前向传播函数

def reset_params(model):  # 重置模型参数的函数
    for i, m in enumerate(model.modules()):  # 对于模型中的每个模块
        weight_init(m)  # 初始化权重

def weight_init(m):  # 权重初始化函数
    if isinstance(m, nn.Conv2d):  # 如果模块是卷积层
        nn.init.xavier_normal_(m.weight)  # 使用Xavier正态分布初始化权重
        nn.init.constant_(m.bias, 0)  # 将偏置初始化为0