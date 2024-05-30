from options.test_options import TestOptions # 导入测试选项
from data import DataLoader  # 导入数据加载器
from models import create_model # 导入模型创建函数
from util.writer import Writer # 导入写入器

# 定义一个运行测试的函数，参数epoch默认为-1
def run_test(epoch=-1):
    print('Running Test') # 打印 "Running Test"
    opt = TestOptions().parse() # 解析测试选项
    opt.serial_batches = True   # 设置为不打乱数据
    dataset = DataLoader(opt)  # 加载数据
    model = create_model(opt)   # 创建模型
    writer = Writer(opt)    # 创建写入器
    # 测试
    writer.reset_counter() # 重置计数器
    for i, data in enumerate(dataset): # 遍历数据集
        model.set_input(data)  # 设置模型输入
        ncorrect, nexamples = model.test()  # 运行模型测试，返回正确的数量和样本的数量
        writer.update_counter(ncorrect, nexamples) # 更新计数器
    writer.print_acc(epoch, writer.acc)  # 打印准确率
    return writer.acc  # 返回准确率

# 如果这个脚本作为主程序运行，那么运行测试
if __name__ == '__main__':
    run_test()
