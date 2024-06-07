import torch
import torch.nn as nn

# 定义一个MeshUnpool类，继承自nn.Module
class MeshUnpool(nn.Module):
    # 初始化函数，设置目标展开大小
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target

    # 定义__call__函数，使得类的实例可以像函数一样被调用
    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    # 定义对组进行填充的函数，如果组的形状小于目标展开大小，则进行填充
    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows =  unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    # 定义对出现次数进行填充的函数，如果出现次数的形状小于目标展开大小，则进行填充
    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    # 定义前向传播函数
    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape
        # 对每个网格的组进行填充
        groups = [self.pad_groups(mesh.get_groups(), edges) for mesh in meshes]
        unroll_mat = torch.cat(groups, dim=0).view(batch_size, edges, -1)
        # 对每个网格的出现次数进行填充
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)
        occurrences = occurrences.expand(unroll_mat.shape)
        unroll_mat = unroll_mat / occurrences
        unroll_mat = unroll_mat.to(features.device)
        # 对每个网格进行展开操作
        for mesh in meshes:
            mesh.unroll_gemm()
        # 返回特征和展开矩阵的乘积
        return torch.matmul(features, unroll_mat)