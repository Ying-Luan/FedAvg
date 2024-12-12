import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Updata import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

matplotlib.use('Agg')  # 配置渲染器


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # 添加 device 信息
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 自定义参数
    args.epochs = 50  # 全体训练轮数
    # args.verbose = True
    args.model = 'cnn'  # 模型
    args.num_channels = 1  # 输入通道数
    args.dataset = 'cifar'  # 数据集
    args.iid = True  # 是否使用 i.i.d

    # 加载数据集和 split users
    if args.dataset == 'mnist':
        # 加载数据集
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # 样例用户
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        # 加载数据集
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../dara/cifar', train=False, download=True, transform=trans_cifar)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # 构造模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)  # 打印模型信息

    net_glob.train()  # 设置模型为训练模式

    # 复制权重
    w_glob = net_glob.state_dict()

    # 训练
    # 初始化数据
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print('Aggregation over all clients')
        w_locals = [w_glob for i in range(args.num_users)]

    # 开始训练
    time_start = time.time()
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 随机选择客户端进行训练
        for idx in idxs_users:
            # 本地训练
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # 存储权重
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))

            # 存储损失
            loss_locals.append(copy.deepcopy(loss))

        # 更新全局权重
        w_glob = FedAvg(w_locals)

        # 复制全局权重到全局模型
        net_glob.load_state_dict(w_glob)

        # 打印损失
        loss_avg = sum(loss_locals) / len(loss_locals)
        time_all = time.time() - time_start
        print('Round {:3d}, Average loss {:.3f}, time {}:{:>2.0f}'.format(iter, loss_avg, int(time_all / 60), time_all % 60))
        loss_train.append(loss_avg)

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid
    ))

    # 测试
    net_glob.eval()  # 设置模型为评估模式
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print('Training accuracy: {:.2f}%'.format(acc_train))
    print('Testing accuracy: {:.2f}%'.format(acc_test))
