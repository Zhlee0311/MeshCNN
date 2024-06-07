import numpy as np  # 导入numpy库，用于进行数值计算
import os  # 导入os库，用于处理文件和目录
import ntpath  # 导入ntpath库，用于处理Windows路径


def fill_mesh(mesh2fill, file: str, opt):
    # 函数用于填充一个网格对象
    # 获取网格文件的路径
    load_path = get_mesh_path(file, opt.num_aug)
    # 如果文件存在，就从文件中加载网格数据
    if os.path.exists(load_path):
        mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
    # 如果文件不存在，就从头开始创建网格数据，并保存到文件中
    else:
        mesh_data = from_scratch(file, opt)
        np.savez_compressed(load_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                            edges_count=mesh_data.edges_count, ve=mesh_data.ve, v_mask=mesh_data.v_mask,
                            filename=mesh_data.filename, sides=mesh_data.sides,
                            edge_lengths=mesh_data.edge_lengths, edge_areas=mesh_data.edge_areas,
                            features=mesh_data.features)
    # 将加载或创建的网格数据填充到mesh2fill对象中
    mesh2fill.vs = mesh_data['vs']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.gemm_edges = mesh_data['gemm_edges']
    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.edge_areas = mesh_data['edge_areas']
    mesh2fill.features = mesh_data['features']
    mesh2fill.sides = mesh_data['sides']

def get_mesh_path(file: str, num_aug: int):
    # 函数用于获取网格文件的路径
    # 获取文件名和扩展名
    filename, _ = os.path.splitext(file)
    # 获取目录名和文件名前缀
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    # 构造加载目录和文件路径
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    # 如果加载目录不存在，就创建它
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    # 返回文件路径
    return load_file

def from_scratch(file, opt):
    # 函数用于从头开始创建网格数据

    class MeshPrep:
        # 定义一个类，用于存储网格数据
        def __getitem__(self, item):
            # 重载索引运算符，使得可以通过属性名获取属性值
            return eval('self.' + item)

    # 创建一个MeshPrep对象，用于存储网格数据
    mesh_data = MeshPrep()
    # 初始化各个属性
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    # 从文件中填充顶点和面
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    # 创建一个全1的布尔数组，用于表示顶点的掩码
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    # 移除非流形面
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    # 如果需要进行数据增强，就进行数据增强
    if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)
    # 构建GEMM结构
    build_gemm(mesh_data, faces, face_areas)
    # 如果需要进行数据增强，就进行后处理
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    # 提取特征
    mesh_data.features = extract_features(mesh_data)
    # 返回网格数据
    return mesh_data

def fill_from_file(mesh, file):
    # 函数用于从文件中填充网格数据
    # 获取文件名和完整文件名
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    # 初始化顶点和面的列表
    vs, faces = [], []
    # 打开文件
    f = open(file)
    # 逐行读取文件
    for line in f:
        # 去除行尾的空白字符
        line = line.strip()
        # 分割行
        splitted_line = line.split()
        # 如果行为空，就跳过
        if not splitted_line:
            continue
        # 如果行的第一个元素是'v'，就将其后面的元素转换为浮点数，添加到顶点列表中
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        # 如果行的第一个元素是'f'，就将其后面的元素转换为整数，添加到面列表中
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            # 断言面的顶点数为3
            assert len(face_vertex_ids) == 3
            # 如果顶点索引为负，就将其转换为正索引
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            # 将面的顶点索引添加到面列表中
            faces.append(face_vertex_ids)
    # 关闭文件
    f.close()
    # 将顶点和面的列表转换为numpy数组
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    # 断言面的顶点索引都在有效范围内
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    # 返回顶点和面
    return vs, faces

def remove_non_manifolds(mesh, faces):
    # 函数用于移除非流形面
    # 初始化每个顶点的边列表
    mesh.ve = [[] for _ in mesh.vs]
    # 初始化边的集合
    edges_set = set()
    # 创建一个全1的布尔数组，用于表示面的掩码
    mask = np.ones(len(faces), dtype=bool)
    # 计算面的法向量和面积
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    # 遍历每个面
    for face_id, face in enumerate(faces):
        # 如果面的面积为0，就将其掩码设置为False，然后跳过
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        # 初始化面的边列表
        faces_edges = []
        # 初始化是否为流形的标志
        is_manifold = False
        # 遍历面的每条边
        for i in range(3):
            # 获取当前边的顶点
            cur_edge = (face[i], face[(i + 1) % 3])
            # 如果当前边已经在边的集合中，就将是否为流形的标志设置为True，然后跳出循环
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                # 否则，将当前边添加到面的边列表中
                faces_edges.append(cur_edge)
        # 如果面是流形的，就将其掩码设置为False
        if is_manifold:
            mask[face_id] = False
        else:
            # 否则，将面的边添加到边的集合中
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    # 返回流形面和其面积
    return faces[mask], face_areas[mask]


def build_gemm(mesh, faces, face_areas):
    # 初始化一些空的列表和字典，用于存储边缘信息和边缘到键的映射。
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []

    # 对于每个面，计算它的所有边缘，并将它们添加到`faces_edges`列表中。
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)

        # 对于`faces_edges`中的每个边缘，如果它还没有在`edge2key`字典中，那么就将它添加到字典中，并将它的键设置为当前的边缘计数。
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1

            # 计算边缘的面积，并将它添加到对应的边缘的面积中。
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3

        # 找到边缘的邻居边缘，并将它们的键添加到`edge_nb`列表中。同时，更新`nb_count`列表。
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2

        # 找到边缘的邻居边缘在`edge_nb`列表中的位置，并将这些位置添加到`sides`列表中。
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2

    # 将所有的列表转换为NumPy数组，并将它们存储在网格对象中。同时，将边缘的面积归一化，使它们的总和为1。
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas)


def compute_face_normals_and_areas(mesh, faces):
    # 计算每个面的法向量和面积
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas

def augmentation(mesh, opt, faces=None):
    # 数据增强方法，包括缩放顶点和翻转边
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces

def post_augmentation(mesh, opt):
    # 数据增强后的处理，包括滑动顶点
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)

def slide_verts(mesh, prct):
    # 滑动顶点的方法，根据二面角和随机数来移动顶点
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)

def scale_verts(mesh, mean=1, var=0.1):
    # 缩放顶点的方法，根据正态分布随机数来缩放顶点
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    # 初始化法向量列表
    normals = [None, None]
    for i in range(2):
        # 计算每个面的两条边
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        # 计算每个面的法向量
        normals[i] = np.cross(edge_a, edge_b)
        # 计算法向量的长度，并进行归一化处理
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    # 计算两个法向量的点积，并将其限制在[-1, 1]范围内
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    # 计算并返回每个面的角度
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    # 获取边的数量，边的面，和边的字典
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    # 计算二面角
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    # 随机排列边的顺序
    edges2flip = np.random.permutation(edge_count)
    # 计算目标翻转的边的数量
    target = int(prct * edge_count)
    flipped = 0
    # 遍历每一条边
    for edge_key in edges2flip:
        # 如果已经翻转的边的数量达到了目标，就停止
        if flipped == target:
            break
        # 如果二面角大于2.7，就尝试翻转这条边
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            # 如果这条边只有一个面，就跳过
            if edge_info[3] == -1:
                continue
            # 计算新的边
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            # 如果新的边已经存在，就跳过
            if new_edge in edges_dict:
                continue
            # 计算新的面
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            # 如果新的面的面积合法，就进行翻转
            if check_area(mesh, new_faces):
                # 删除旧的边
                del edges_dict[(edge_info[0], edge_info[1])]
                # 更新边的信息
                edge_info[:2] = [new_edge[0], new_edge[1]]
                # 添加新的边
                edges_dict[new_edge] = edge_key
                # 重建面
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                # 更新面的邻居信息
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                # 增加已经翻转的边的数量
                flipped += 1
    # 返回更新后的面
    return faces


def rebuild_face(face, new_face):
    # 计算新面和旧面的差集，得到新的点
    new_point = list(set(new_face) - set(face))[0]
    # 遍历旧面的每个点
    for i in range(3):
        # 如果旧面的点不在新面中，就用新的点替换它
        if face[i] not in new_face:
            face[i] = new_point
            break
    # 返回更新后的面
    return face

def check_area(mesh, faces):
    # 计算面的法向量
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    # 计算面的面积
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    # 检查每个面的面积是否大于0
    return face_areas[0] > 0 and face_areas[1] > 0

def get_edge_faces(faces):
    # 初始化边的数量
    edge_count = 0
    # 初始化边的面列表
    edge_faces = []
    # 初始化边的字典
    edge2keys = dict()
    # 遍历每个面
    for face_id, face in enumerate(faces):
        # 遍历每个面的每条边
        for i in range(3):
            # 计算当前边
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            # 如果当前边不在边的字典中，就添加它
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            # 获取当前边的键
            edge_key = edge2keys[cur_edge]
            # 如果当前边的第一个面还没有被设置，就设置它
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            # 否则，设置当前边的第二个面
            else:
                edge_faces[edge_key][3] = face_id
    # 返回边的数量，边的面，和边的字典
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    # 如果没有提供边的点，就获取它们
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    # 计算每条边的长度
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    # 设置网格的边的长度
    mesh.edge_lengths = edge_lengths

def extract_features(mesh):
    # 初始化特征列表
    features = []
    # 获取边的点
    edge_points = get_edge_points(mesh)
    # 设置边的长度
    set_edge_lengths(mesh, edge_points)
    # 设置错误状态，当除法出现错误时抛出异常
    with np.errstate(divide='raise'):
        try:
            # 遍历每个特征提取器
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                # 提取特征
                feature = extractor(mesh, edge_points)
                # 添加特征到特征列表
                features.append(feature)
            # 将所有的特征连接成一个数组
            return np.concatenate(features, axis=0)
        except Exception as e:
            # 打印异常信息
            print(e)
            # 抛出值错误，表示特征提取失败
            raise ValueError(mesh.filename, 'bad features')

def dihedral_angle(mesh, edge_points):
    # 获取第一个面的法向量
    normals_a = get_normals(mesh, edge_points, 0)
    # 获取第二个面的法向量
    normals_b = get_normals(mesh, edge_points, 3)
    # 计算两个法向量的点积，并将结果限制在[-1, 1]之间
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    # 计算二面角
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    # 返回二面角
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ 计算两个角度：每个边共享的每个面一个
        角度在每个面对边的边上
        排序处理顺序的模糊性
    """
    # 获取第一个面的对边角
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    # 获取第二个面的对边角
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    # 将两个角度连接成一个数组
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    # 对角度进行排序
    angles = np.sort(angles, axis=0)
    # 返回角度
    return angles


def symmetric_ratios(mesh, edge_points):
    """ 计算两个比例：每个边共享的每个面一个
        比例是每个三角形的高度 / 底边（边）
        排序处理顺序的模糊性
    """
    # 获取第一个面的比例
    ratios_a = get_ratios(mesh, edge_points, 0)
    # 获取第二个面的比例
    ratios_b = get_ratios(mesh, edge_points, 3)
    # 将两个比例连接成一个数组
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    # 对比例进行排序
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    """ 返回：edge_points (#E x 4) 张量，每条边有四个顶点的id
        例如：edge_points[edge_id, 0] 和 edge_points[edge_id, 1] 是定义 edge_id 的两个顶点
        edge_id 的每个相邻面有另一个顶点，即 edge_points[edge_id, 2] 或 edge_points[edge_id, 3]
    """
    # 初始化边的点
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    # 遍历每条边
    for edge_id, edge in enumerate(mesh.edges):
        # 获取边的点
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    # 返回边的点
    return edge_points


def get_side_points(mesh, edge_id):
    # 获取指定边
    edge_a = mesh.edges[edge_id]

    # 根据gemm_edges的值，获取与指定边相邻的两条边
    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]

    # 根据gemm_edges的值，获取与指定边相邻的另外两条边
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]

    # 初始化三个顶点的索引
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0

    # 根据边的顶点是否在其他边中，确定顶点的索引
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1

    # 返回四个顶点
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    # 计算两个向量，它们分别是从边的一个顶点到另外两个顶点的向量
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    # 计算两个向量的叉积，得到法线
    normals = np.cross(edge_a, edge_b)
    # 计算法线的长度，并进行固定除法，防止除数为0
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    # 将法线除以其长度，得到单位法线
    normals /= div[:, np.newaxis]
    # 返回单位法线
    return normals

def get_opposite_angles(mesh, edge_points, side):
    # 计算两个向量，它们分别是从边的一个顶点到另外两个顶点的向量
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    # 将两个向量除以其长度，得到单位向量
    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    # 计算两个单位向量的点积，并将其限制在[-1, 1]范围内
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    # 返回两个单位向量的夹角
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    # 计算边的长度
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]], ord=2, axis=1)
    # 获取三个点的坐标
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    # 计算向量AB
    line_ab = point_b - point_a
    # 计算点O到线段AB的投影长度
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    # 计算投影点的坐标
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    # 计算点O到投影点的距离
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    # 返回距离与边长的比率
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    # 如果epsilon为0，将to_div中为0的元素替换为0.1
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    # 否则，将to_div中的每个元素加上epsilon
    else:
        to_div += epsilon
    # 返回结果
    return to_div
