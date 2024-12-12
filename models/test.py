import torch


def test_img(net_g, dataset, args):
    net_g.eval()  # 设置模型为评估模式

    # 测试
    test_loss = 0
    correct = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)

        # 将每个 batch 的损失求和
        test_loss += torch.nn.functional.cross_entropy(log_probs, target, reduction='sum').item()

        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f}\nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy
        ))

    return accuracy, test_loss
