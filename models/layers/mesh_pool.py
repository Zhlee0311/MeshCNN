import torch
# 导入torch库，这是一个用于处理张量计算和深度学习的库

import torch.nn as nn
# 导入torch.nn模块，这是一个用于构建神经网络的模块

from threading import Thread
# 导入Thread，这是一个用于创建和管理线程的类

from models.layers.mesh_union import MeshUnion
# 导入MeshUnion，这是一个用于处理网格合并的类

import numpy as np
# 导入numpy库，这是一个用于处理多维数组和矩阵运算的库

from heapq import heappop, heapify
# 导入heappop和heapify，这是两个用于处理堆数据结构的函数


class MeshPool(nn.Module):
    def __init__(self, target, multi_thread=False):
        # 初始化函数，设置目标输出大小，是否使用多线程，并初始化一些变量
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        # 当实例被当作函数调用时，调用forward方法
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        # 前向传播函数，输入特征和网格，输出池化后的特征
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # 遍历每个网格
        for mesh_index in range(len(meshes)):
            # 如果使用多线程，则创建新线程进行池化操作
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                # 否则在主线程中进行池化操作
                self.__pool_main(mesh_index)
        # 如果使用多线程，则等待所有线程完成
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        # 将所有网格的特征拼接起来，并调整形状
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        # 主要的池化函数，对指定的网格进行池化操作
        # 获取指定的网格
        mesh = self.__meshes[mesh_index]
        # 构建优先队列，队列中的元素是边的id和对应的特征
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count], mesh.edges_count)
        # 创建一个全为True的掩码，用于标记哪些边还没有被池化
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        # 创建一个MeshUnion对象，用于管理边的组合
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
         # 当网格的边数大于目标边数时，进行循环
        while mesh.edges_count > self.__out_target:
            # 从队列中取出一个边
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            # 如果这个边还没有被池化，那么进行池化操作
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        # 清理网格，删除已经被池化的边
        mesh.clean(mask, edge_groups)
        # 重新构建特征
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        # 更新特征
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
         # 池化一个边，如果边在边界上，或者边的两侧有无效的边，或者边的一环不合法，那么这个边不能被池化
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0)\
            and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
            and self.__is_one_ring_valid(mesh, edge_id):
            # 如果边可以被池化，那么进行池化操作
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            # 合并边的两个顶点
            mesh.merge_vertices(edge_id)
            # 标记这个边已经被池化
            mask[edge_id] = False
            # 从组中删除这个边
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            # 边的数量减一
            mesh.edges_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        # 清理一个边的一侧，如果这一侧有无效的边，那么删除这些边
        # 如果网格的边数已经小于或等于目标边数，那么直接返回False
        if mesh.edges_count <= self.__out_target:
            return False
        # 获取无效的边
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        # 当无效的边的数量不为0，并且网格的边数大于目标边数时，进行循环
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            # 删除无效的边
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            # 如果网格的边数已经小于或等于目标边数，那么直接返回False
            if mesh.edges_count <= self.__out_target:
                return False
            # 如果边在边界上，那么直接返回False
            if self.has_boundaries(mesh, edge_id):
                return False
            # 重新获取无效的边
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        # 如果循环结束，那么返回True，表示这一侧的边已经被清理完毕
        return True
    
    
    @staticmethod
    def has_boundaries(mesh, edge_id):
        # 静态方法，判断一个边是否在边界上
        # 遍历指定边的所有邻边
        for edge in mesh.gemm_edges[edge_id]:
            # 如果邻边的id为-1，或者邻边的邻边中有-1，那么说明这个边在边界上
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        # 如果所有的邻边都不在边界上，那么这个边也不在边界上
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        # 静态方法，判断一个边的一环是否有效
        # 获取边的两个顶点的所有边
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        # 计算两个顶点的边的交集，然后去掉这个边本身
        shared = v_a & v_b - set(mesh.edges[edge_id])
        # 如果交集的大小为2，那么这个边的一环是有效的
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        # 对一个边的一侧进行池化操作
        # 获取边的面信息
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        # 重定向边
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        # 合并组
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        # 标记key_b为False，表示这个边已经被池化
        mask[key_b] = False
        # 移除组
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        # 移除边
        mesh.remove_edge(key_b)
        # 边的数量减1
        mesh.edges_count -= 1
        # 返回key_a，表示这个边已经被池化
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        # 静态方法，获取无效的边
        # 获取边的面信息
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        # 获取共享的元素
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        # 如果没有共享的元素，那么返回空列表
        if len(shared_items) == 0:
            return []
        else:
            # 如果有共享的元素，那么应该有两个
            assert (len(shared_items) == 2)
            # 获取中间的边
            middle_edge = other_keys_a[shared_items[0]]
            # 获取更新的key和side
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            # 重定向边
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b, MeshPool.__get_other_side(update_side_b))
            # 合并组
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
            # 返回无效的边
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        # 静态方法，重定向边
        # 更新mesh的gemm_edges和sides属性，使得边A的一侧指向边B，边B的一侧指向边A
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        # 静态方法，获取两个列表中的共享元素
        # 遍历两个列表，如果发现相同的元素，就将它们的索引添加到shared_items列表中
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        # 静态方法，获取边的另一侧
        # 如果side是偶数，那么返回side+1，如果side是奇数，那么返回side-1
        return side + 1 - 2 * (side % 2)    

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        # 静态方法，获取一个边的面信息
        # 获取边的两个关键边和两个侧面
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        # 计算另外两个侧面
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        # 获取另外两个侧面的关键边
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        # 返回所有的信息
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        # 静态方法，移除一个三元组
        # 获取三元组的顶点
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            # 更新顶点集合，使其只包含所有边共有的顶点
            vertex &= set(mesh.edges[edge_key])
            # 标记边为已移除
            mask[edge_key] = False
            # 从边组中移除边
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        # 更新边的数量
        mesh.edges_count -= 3
        # 确保只有一个顶点
        vertex = list(vertex)
        assert(len(vertex) == 1)
        # 从mesh中移除顶点
        mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, edges_count):
        # 构建一个优先队列，用于删除具有最小范数的边
        # 计算特征的平方和，得到每个边的范数的平方
        quared_magnitude = torch.sum(features * features, 0)
        # 如果squared_magnitude的最后一个维度不是1，那么增加一个维度
        if squared_magnitude.shape[-1] != 1:
             squared_magnitude = squared_magnitude.unsqueeze(-1)
        # 创建一个从0到edges_count的序列，这个序列代表每个边的id
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        # 将squared_magnitude和edge_ids拼接起来，得到一个堆
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        # 使用heapify函数，将列表转换为堆
        heapify(heap)
        # 返回构建的堆
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        # 静态方法，合并两个组
        # 在edge_groups和mesh中，将source组和target组合并
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        # 静态方法，移除一个组
        # 在edge_groups和mesh中，移除指定索引的组
        edge_groups.remove_group(index)
        mesh.remove_group(index)

