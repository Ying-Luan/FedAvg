import copy
import torch


def FedAvg(w):
    """
    将客户端的权重求平均
    :param w: 初始各客户端的权重
    :return: 平均权重
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # 对于每一个权重
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))  # 逐元素除法
    return w_avg
