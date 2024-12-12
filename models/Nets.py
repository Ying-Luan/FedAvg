import torch


class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.layer_input = torch.nn.Linear(dim_in, dim_hidden)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer_hidden = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)  # batch * 1 * 28 * 28 -> batch * 10 * 24 * 24
        x = torch.nn.functional.max_pool2d(x, 2)  # batch * 10 * 24 * 24 -> batch * 10 * 12 * 12
        x = torch.nn.functional.relu(x)  # 激活
        x = self.conv2(x)  # batch * 10 * 12 * 12 -> batch * 20 * 8 * 8
        x = self.conv2_drop(x)  # 随机丢弃
        x = torch.nn.functional.max_pool2d(x, 2)  # batch * 20 * 8 * 8 -> batch * 20 * 4 * 4
        x = torch.nn.functional.relu(x)  # 激活
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])  # batch * 20 * 4 * 4 -> batch * 320
        x = self.fc1(x)  # batch * 320 -> batch * 50
        x = torch.nn.functional.relu(x)  # 激活
        x = torch.nn.functional.dropout(x, training=self.training)  # 随机丢弃
        x = self.fc2(x)  # batch * 50 -> batch * 10
        return x


class CNNCifar(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)  # batch * 3 * 32 * 32 -> batch * 6 * 28 * 28
        x = torch.nn.functional.relu(x)  # 激活
        x = self.pool(x)  # batch * 6 * 28 * 28 -> batch * 6 * 14 * 14
        x = self.conv2(x)  # batch * 6 * 14 * 14 -> batch * 16 * 10 * 10
        x = torch.nn.functional.relu(x)  # 激活
        x = self.pool(x)  # batch * 16 * 10 * 10 -> batch * 16 * 5 * 5
        x = x.view(-1, 16 * 5 * 5)  # batch * 16 * 5 * 5 -> batch * 400
        x = self.fc1(x)  # batch * 400 -> batch * 120
        x = torch.nn.functional.relu(x)  # 激活
        x = self.fc2(x)  # batch * 120 -> batch * 84
        x = torch.nn.functional.relu(x)  # 激活
        x = self.fc3(x)  # batch * 84 -> batch * 10
        return x
