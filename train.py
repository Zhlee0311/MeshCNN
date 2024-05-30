import time # 导入时间模块
from options.train_options import TrainOptions # 导入训练选项
from data import DataLoader # 导入数据加载器
from models import create_model  # 导入模型创建函数
from util.writer import Writer   # 导入写入器
from test import run_test     # 导入测试函数

if __name__ == '__main__':
    opt = TrainOptions().parse()  # 解析训练选项
    dataset = DataLoader(opt)    # 使用选项创建数据加载器
    dataset_size = len(dataset)   # 获取数据集的大小
    print('#training meshes = %d' % dataset_size) # 打印训练网格的数量

    model = create_model(opt) # 使用选项创建模型
    writer = Writer(opt)   # 使用选项创建写入器
    total_steps = 0       # 初始化总步数为0

     # 遍历每个训练周期
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()   # 记录训练周期开始的时间
        iter_data_time = time.time()    # 记录数据迭代的时间
        epoch_iter = 0        # 初始化训练周期迭代次数为0

         # 遍历数据集中的每个数据
        for i, data in enumerate(dataset):
            iter_start_time = time.time() # 记录迭代开始的时间
              # 如果总步数是打印频率的倍数，记录数据迭代的时间
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size # 更新总步数
            epoch_iter += opt.batch_size  # 更新训练周期迭代次数
            model.set_input(data)  # 设置模型输入
            model.optimize_parameters()  # 优化模型参数

             # 如果总步数是打印频率的倍数，打印当前的损失
            if total_steps % opt.print_freq == 0:
                loss = model.loss # 获取模型损失
                t = (time.time() - iter_start_time) / opt.batch_size # 计算每个批次的时间
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)  # 打印当前的损失
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size) # 绘制损失图

            # 如果数据索引是保存最新模型频率的倍数，保存最新的模型
            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))   # 打印保存最新模型的信息
                model.save_network('latest')  # 保存最新的网络

            iter_data_time = time.time()  # 更新数据迭代的时间

        # 如果训练周期是保存模型频率的倍数，保存模型    
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))  # 打印保存模型的信息
            model.save_network('latest') # 保存最新的网络
            model.save_network(epoch)  #保存当前训练周期的网络

        # 打印训练周期结束的信息
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate() # 更新学习率
        
        # 如果选项中设置了详细绘图，绘制模型权重图
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        # 如果训练周期是运行测试频率的倍数，运行测试
        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch) # 运行测试，获取准确率
            writer.plot_acc(acc, epoch) # 绘制准确率图

    writer.close()  # 关闭写入器
