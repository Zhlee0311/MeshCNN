import torch # 导入PyTorch库
from . import networks # 导入当前目录下的networks模块
from os.path import join # 导入os.path模块的join函数
from util.util import seg_accuracy, print_network # 导入util模块的seg_accuracy和print_network函数


class ClassifierModel:
    """ 
    用于训练模型权重的类

    :args opt: 包含配置参数的结构
     例如，
    --dataset_mode -> 分类 / 分割
    --arch -> 网络类型
    """
    def __init__(self, opt):
        self.opt = opt # 存储配置参数
        self.gpu_ids = opt.gpu_ids # 存储GPU ID
        self.is_train = opt.is_train # 存储训练状态
        # 根据GPU ID设置设备，如果没有GPU ID，则使用CPU
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name) # 设置保存目录
        self.optimizer = None  # 初始化优化器
        self.edge_features = None # 初始化边缘特征
        self.labels = None # 初始化标签
        self.mesh = None # 初始化网格
        self.soft_label = None # 初始化软标签
        self.loss = None  # 初始化损失

        
        self.nclasses = opt.nclasses # 存储类别数量

        # 加载/定义网络
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train) # 设置网络的训练状态
        self.criterion = networks.define_loss(opt).to(self.device) # 定义损失函数并将其移动到指定的设备

        if self.is_train: # 如果处于训练状态
             # 使用Adam优化器，学习率为opt.lr，beta参数为(opt.beta1, 0.999)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt) # 获取优化器的调度器
            print_network(self.net)  # 打印网络结构

        if not self.is_train or opt.continue_train: # 如果处于非训练状态或设置为继续训练
            self.load_network(opt.which_epoch)  # 加载指定周期的网络

    def set_input(self, data): # 设置输入数据
        input_edge_features = torch.from_numpy(data['edge_features']).float() # 获取边缘特征并转换为浮点类型
        labels = torch.from_numpy(data['label']).long() # 获取标签并转换为长整型
         # 将输入数据和标签移动到指定的设备，并根据训练状态设置是否需要计算梯度
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']  # 获取网格数据
        # 如果数据集模式为'segmentation'且处于非训练状态，获取软标签
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):  # 前向传播
        # 将边缘特征和网格数据传递给网络并获取输出
        out = self.net(self.edge_features, self.mesh)
        return out # 返回网络输出


    def backward(self, out): # 反向传播
        self.loss = self.criterion(out, self.labels) # 计算损失
        self.loss.backward() # 反向传播损失

    def optimize_parameters(self):  # 优化参数
        self.optimizer.zero_grad()  # 清零优化器的梯度
        out = self.forward() # 前向传播
        self.backward(out)  # 反向传播
        self.optimizer.step()  # 更新参数



##################

    def load_network(self, which_epoch):  # 从磁盘加载模型
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch  # 定义保存的文件名
        load_path = join(self.save_dir, save_filename) # 定义加载路径
        net = self.net
        if isinstance(net, torch.nn.DataParallel):  # 如果网络是DataParallel类型，获取其内部的模块
            net = net.module
        print('loading the model from %s' % load_path) # 打印加载信息
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device)) # 加载模型状态
        if hasattr(state_dict, '_metadata'):  # 如果状态字典有'_metadata'属性，删除它
            del state_dict._metadata
        net.load_state_dict(state_dict)  # 加载模型状态


    def save_network(self, which_epoch):  # 将模型保存到磁盘
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch) # 定义保存的文件名
        save_path = join(self.save_dir, save_filename)  # 定义保存路径
        if len(self.gpu_ids) > 0 and torch.cuda.is_available(): # 如果有GPU ID且CUDA可用
            torch.save(self.net.module.cpu().state_dict(), save_path)   # 将模型状态保存到磁盘
            self.net.cuda(self.gpu_ids[0])       # 将模型移动到GPU
        else: # 如果没有可用的GPU
            torch.save(self.net.cpu().state_dict(), save_path)  # 将模型状态保存到磁盘

    def update_learning_rate(self):  # 更新学习率（每个周期调用一次）
        """update learning rate (called once every epoch)"""
        self.scheduler.step()  # 调度器步进
        lr = self.optimizer.param_groups[0]['lr'] # 获取当前的学习率
        print('learning rate = %.7f' % lr)  # 打印当前的学习率


    def test(self): # 测试模型
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad(): # 不计算梯度
            out = self.forward()   # 前向传播
            pred_class = out.data.max(1)[1] # 获取预测的类别
            label_class = self.labels # 获取标签的类别
            self.export_segmentation(pred_class.cpu()) # 导出分割结果
            correct = self.get_accuracy(pred_class, label_class) # 计算准确率
        return correct, len(label_class)  # 返回正确的数量和总数量

    def get_accuracy(self, pred, labels):  # 计算准确率
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification': # 如果数据集模式为'classification'
            correct = pred.eq(labels).sum() # 计算分类的准确率
        elif self.opt.dataset_mode == 'segmentation':  # 如果数据集模式为'segmentation'
            correct = seg_accuracy(pred, self.soft_label, self.mesh)  # 计算分割的准确率
        return correct  # 返回准确率

    def export_segmentation(self, pred_seg):  # 导出分割结果    
        if self.opt.dataset_mode == 'segmentation': # 如果数据集模式为'segmentation'
            for meshi, mesh in enumerate(self.mesh):  # 遍历每个网格
                mesh.export_segments(pred_seg[meshi, :])  # 导出分割结果
