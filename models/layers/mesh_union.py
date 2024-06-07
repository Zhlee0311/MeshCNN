import torch # 导入torch库，这是一个用于处理张量计算和深度学习的库
from torch.nn import ConstantPad2d # 导入ConstantPad2d，这是一个用于对二维输入进行常数填充的函数

class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        # 初始化函数，设置大小和设备，并设置重建特征的方法为平均重建
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

    def union(self, source, target):
        # 合并两个组
        self.groups[target, :] += self.groups[source, :]

    def remove_group(self, index):
        # 移除组，此函数未实现
        return

    def get_group(self, edge_key):
        # 获取组
        return self.groups[edge_key, :]

    def get_occurrences(self):
        # 获取出现次数
        return torch.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        # 获取组，使用张量掩码
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]
    
    def rebuild_features_average(self, features, mask, target_edges):
        # 准备组
        self.prepare_groups(features, mask)
        # 计算特征和组的矩阵乘积
        fe = torch.matmul(features.squeeze(-1), self.groups)
        # 计算每个组的出现次数
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        # 计算平均特征
        fe = fe / occurrences
        # 计算需要填充的数量
        padding_b = target_edges - fe.shape[1]
        # 如果需要填充，使用ConstantPad2d进行填充
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        # 返回特征
        return fe

def prepare_groups(self, features, mask):
        # 将掩码转换为张量
        tensor_mask = torch.from_numpy(mask)
        # 获取需要的组，并进行转置
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)
        # 计算需要填充的数量
        padding_a = features.shape[1] - self.groups.shape[0]
        # 如果需要填充，使用ConstantPad2d进行填充
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
