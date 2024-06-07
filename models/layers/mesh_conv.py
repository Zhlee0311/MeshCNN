import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshConv(nn.Module):
    """ 
    计算边和4个邻接（1环）边邻居之间的卷积
    在前向传播中接收：
    x: 边特征 (Batch x Features x Edges)
    mesh: mesh数据结构的列表 (len(mesh) == Batch)
    并应用卷积
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()  # 调用父类的初始化函数
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)  # 创建卷积层
        self.k = k  # 卷积核大小

    def __call__(self, edge_f, mesh):
        # 定义__call__方法，使得对象可以像函数一样被调用
        return self.forward(edge_f, mesh)  # 调用前向传播函数

    def forward(self, x, mesh):
        # 定义前向传播函数
        x = x.squeeze(-1)  # 去掉最后一个维度
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)  # 对每个mesh进行pad_gemm操作，并将结果拼接
        # 构建'邻域图像'并应用卷积
        G = self.create_GeMM(x, G)  # 创建GeMM
        x = self.conv(G)  # 对G进行卷积操作
        return x  # 返回结果

    def flatten_gemm_inds(self, Gi):
        # 将Gi展平
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # 展平Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ 
        使用1环索引（Gi）收集边特征（x）
        应用对称函数来处理顺序不变性
        返回一个可以使用2d卷积的'假图像'
        输出维度：Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # 用零填充每个批次样本的第一行
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift

        # 首先展平索引
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # 应用对称函数进行等变卷积
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ 
        提取一环邻居（4x）-> m.gemm_edges
        其大小为 #edges x 4
        添加边本身的id使其变为 #edges x 5
        然后填充到所需的大小，例如，xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # 使用F进行填充
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm