import torch
import numpy as np


class DataserSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = torch.nn.CrossEntropyLoss()  # 损失函数
        self.selected_client = []
        self.ldr_train = torch.utils.data.DataLoader(DataserSplit(dataset, idxs),
                                                     batch_size=self.args.local_bs,
                                                     shuffle=True
                                                     )

    def train(self, net):
        """
        本地训练模型
        :param net:
        :return: 模型权重, 损失
        """
        net.train()  # 设置模型为训练模式

        # 训练和更新
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()  # 梯度清零
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                # 是否显示训练信息
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                        100.*batch_idx / len(self.ldr_train), loss.item()
                    ))

                batch_loss.append(loss.item())  # 记录损失

            epoch_loss.append(sum(batch_loss)/len(batch_loss))  # 记录每一轮的损失

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
