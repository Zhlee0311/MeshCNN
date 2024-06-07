from tempfile import mkstemp  # 用于创建临时文件
from shutil import move  # 用于执行高级文件操作，如复制和删除文件
import torch  # PyTorch库，用于构建和训练神经网络
import numpy as np  # 用于科学计算的库
import os  # 用于操作系统相关的操作，如读/写文件
from models.layers.mesh_union import MeshUnion  # 从mesh_union模块导入MeshUnion类
from models.layers.mesh_prepare import fill_mesh  # 从mesh_prepare模块导入fill_mesh函数


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        # 初始化各种属性
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        fill_mesh(self, file, opt)  # 填充网格
        self.export_folder = export_folder  # 导出文件夹
        self.history_data = None  # 历史数据
        if hold_history:  # 如果需要保持历史记录
            self.init_history()  # 初始化历史记录
        self.export()  # 导出

    def extract_features(self):
        # 提取特征
        return self.features

    def merge_vertices(self, edge_id):
        # 合并顶点
        self.remove_edge(edge_id)  # 移除边
        edge = self.edges[edge_id]  # 获取边
        v_a = self.vs[edge[0]]  # 获取顶点A
        v_b = self.vs[edge[1]]  # 获取顶点B
        # 更新pA
        v_a.__iadd__(v_b)  # 将B加到A上
        v_a.__itruediv__(2)  # 将A除以2
        self.v_mask[edge[1]] = False  # 设置顶点B的掩码为False
        mask = self.edges == edge[1]  # 创建掩码
        self.ve[edge[0]].extend(self.ve[edge[1]])  # 将顶点B的边扩展到顶点A上
        self.edges[mask] = edge[0]  # 更新边

    def remove_vertex(self, v):
        # 移除顶点
        self.v_mask[v] = False  # 设置顶点的掩码为False

    def remove_edge(self, edge_id):
        # 移除边
        vs = self.edges[edge_id]  # 获取边的顶点
        for v in vs:  # 对于每个顶点
            if edge_id not in self.ve[v]:  # 如果边不在顶点的边列表中
                print(self.ve[v])  # 打印顶点的边列表
                print(self.filename)  # 打印文件名
            self.ve[v].remove(edge_id)  # 从顶点的边列表中移除边

    def clean(self, edges_mask, groups):
        # 将edges_mask转换为布尔类型
        edges_mask = edges_mask.astype(bool)
        # 将edges_mask复制并转换为torch张量
        torch_mask = torch.from_numpy(edges_mask.copy())
        # 使用edges_mask更新gemm_edges、edges和sides
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        # 初始化新的顶点-边列表
        new_ve = []
        # 在edges_mask后面添加一个False
        edges_mask = np.concatenate([edges_mask, [False]])
        # 初始化新的索引数组，所有元素都为0，长度与edges_mask相同
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        # 将最后一个元素设置为-1
        new_indices[-1] = -1
        # 将edges_mask中为True的位置设置为0到edges_mask中True的数量
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        # 使用新的索引更新gemm_edges
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        # 遍历ve列表
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # 遍历ve中的每个元素
            for e in ve:
                # 将新的索引添加到update_ve中
                update_ve.append(new_indices[e])
            # 将update_ve添加到new_ve中
            new_ve.append(update_ve)
        # 更新ve
        self.ve = new_ve
        # 清理历史记录
        self.__clean_history(groups, torch_mask)
        # 更新pool_count
        self.pool_count += 1
        # 导出
        self.export()

    def export(self, file=None, vcolor=None):
        # 如果没有提供文件名
        if file is None:
            # 如果有导出文件夹
            if self.export_folder:
                # 分割文件名和扩展名
                filename, file_extension = os.path.splitext(self.filename)
                # 创建新的文件名
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                # 如果没有导出文件夹，直接返回
                return
        # 初始化面列表
        faces = []
        # 获取顶点
        vs = self.vs[self.v_mask]
        # 获取gemm边
        gemm = np.array(self.gemm_edges)
        # 创建新的索引数组
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        # 更新新的索引数组
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        # 遍历gemm边
        for edge_index in range(len(gemm)):
            # 获取循环
            cycles = self.__get_cycle(gemm, edge_index)
            # 遍历循环
            for cycle in cycles:
                # 将循环转换为面并添加到面列表中
                faces.append(self.__cycle_to_face(cycle, new_indices))
        # 打开文件
        with open(file, 'w+') as f:
            # 遍历顶点
            for vi, v in enumerate(vs):
                # 如果有顶点颜色，获取顶点颜色
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                # 写入顶点
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            # 遍历面
            for face_id in range(len(faces) - 1):
                # 写入面
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            # 写入最后一个面
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            # 遍历边
            for edge in self.edges:
                # 写入边
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        # 如果没有导出文件夹，直接返回
        if not self.export_folder:
            return
        # 获取当前的分段
        cur_segments = segments
        # 遍历所有的池
        for i in range(self.pool_count + 1):
            # 分割文件名和扩展名
            filename, file_extension = os.path.splitext(self.filename)
            # 创建新的文件名
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            # 创建临时文件
            fh, abs_path = mkstemp()
            # 初始化边键
            edge_key = 0
            # 打开新文件
            with os.fdopen(fh, 'w') as new_file:
                # 打开旧文件
                with open(file) as old_file:
                    # 遍历旧文件的每一行
                    for line in old_file:
                        # 如果行的第一个字符是'e'
                        if line[0] == 'e':
                            # 写入新的行，包含边和对应的分段
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            # 如果边键小于分段的长度
                            if edge_key < len(cur_segments):
                                # 边键加1
                                edge_key += 1
                                # 写入换行符
                                new_file.write('\n')
                        else:
                            # 如果行的第一个字符不是'e'，直接写入行
                            new_file.write(line)
            # 删除旧文件
            os.remove(file)
            # 将临时文件移动到旧文件的位置
            move(abs_path, file)
            # 如果i小于历史数据中边掩码的长度
            if i < len(self.history_data['edges_mask']):
                # 更新当前的分段
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        # 初始化循环列表
        cycles = []
        # 遍历两次
        for j in range(2):
            # 初始化下一个边和起始点
            next_side = start_point = j * 2
            next_key = edge_id
            # 如果gemm中对应的元素为-1，跳过当前循环
            if gemm[edge_id, start_point] == -1:
                continue
            # 添加新的循环
            cycles.append([])
            # 遍历三次
            for i in range(3):
                # 获取下一个键和边
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                # 更新gemm中对应的元素为-1
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                # 更新下一个键和边
                next_key = tmp_next_key
                next_side = tmp_next_side
                # 添加下一个键到最后一个循环中
                cycles[-1].append(next_key)
        # 返回循环列表

    def __cycle_to_face(self, cycle, v_indices):
        # 初始化面列表
        face = []
        # 遍历三次
        for i in range(3):
            # 获取顶点
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            # 添加顶点索引到面列表中
            face.append(v_indices[v])
        # 返回面列表

    def init_history(self):
        # 初始化历史数据
        self.history_data = {
                               'groups': [],
                               'gemm_edges': [self.gemm_edges.copy()],
                               'occurrences': [],
                               'old2current': np.arange(self.edges_count, dtype=np.int32),
                               'current2old': np.arange(self.edges_count, dtype=np.int32),
                               'edges_mask': [torch.ones(self.edges_count,dtype=torch.bool)],
                               'edges_count': [self.edges_count],
                              }
        # 如果有导出文件夹
        if self.export_folder:
            # 添加'collapses'到历史数据中
            self.history_data['collapses'] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        # 如果存在导出文件夹和历史数据
        if self.export_folder and self.history_data:
            # 合并两个组
            self.history_data['collapses'].union(self.history_data['current2old'][source], self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        # 如果存在历史数据
        if self.history_data is not None:
            # 移除组
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            # 如果存在导出文件夹
            if self.export_folder:
                # 移除组
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        # 获取组
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        # 获取出现次数
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        # 如果存在历史数据
        if self.history_data is not None:
            # 清理历史数据
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            # 如果存在导出文件夹
            if self.export_folder != '':
                # 添加边掩码
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            # 添加出现次数和组
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            # 添加gemm边和边数
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
    
    def unroll_gemm(self):
        # 取消gemm边的滚动
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        # 获取边的区域
        return self.edge_areas
