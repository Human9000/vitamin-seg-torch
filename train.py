import numpy as np
import torch
from tqdm import tqdm

from network.segvitamin import SegVitamin
from dataset._dataset import MyDataset, DataLoader

# 下面是一系列配置参数
epochs = 10000
batch_size = 4
weights_path = f"weights/SegVitamin.pth"

# 下面的内容可以修改为适合你的任务的损失函数
def loss_func(pred, gt): 
    # 由于背景过多，采用选择器均衡一下
    select1 = 1. * (gt == 1)
    select2 = 1. * (0.3 <= gt) * (gt < 1)
    select3 = 1. * (gt < 0.3)

    # 方差误差
    err = (p - gt).pow(2)

    # 均衡的均方差
    loss1 = (err * select1).sum() / (select1.sum() + 1)
    loss2 = (err * select2).sum() / (select2.sum() + 1)
    loss3 = (err * select3).sum() / (select3.sum() + 1)

    loss =  loss1 + loss2 + loss3
    return loss

    
if __name__ == '__main__':
    my_dataset = MyDataset(False)  # 获取 10s 的数据
    loader = DataLoader(my_dataset, batch_size, True, num_workers=2)
    net = SegVitamin(size=(1024, 1536), fact=(2, 2), in_chans=1, out_channel=13)
    net.train()
    net.cuda()
    try:
        save_data = torch.load(weights_path) 
    except:
        print('No weights found, starting from scratch')
        save_data = {'loss_s': [1e9],
                     'epoch_s': [-1],
                     'best_loss': 1e9,
                     'best_epoch': -1,
                     'best_weights': net.state_dict(),
                     'last_weights': net.state_dict()}
    state_dict = save_data['last_weights']
    net.load_state_dict(state_dict, strict=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-9)
    h = np.linspace(np.zeros(1000),
                    np.ones(1000),
                    100).T
    h = torch.from_numpy(h).cuda().float()[None, None]
    init_epoch = save_data['epoch_s'][-1] + 1
    loader.dataset.t = 10
    for epoch in range(init_epoch, init_epoch + epochs):
        avgloss = 0
        n = 0
        with tqdm(loader, total=200) as tbr:
            for mask, feature in tbr:
                opt.zero_grad()
                mask = mask.cuda().float()
                feature = feature.cuda().float()
                p = net(mask) 
                loss =  loss_func(p, feature) 
                loss.backward()
                opt.step()
                avgloss += loss.item()
                n += 1
                tbr.set_description(f"train loss={loss.item()}")
                lr_scheduler.step()
                if n>=200:
                    break

        avgloss = avgloss / n
        print(f"epoch {epoch}: avg_loss={avgloss} avg_loss_0.5={avgloss ** 0.5}")
        save_data['epoch_s'].append(save_data['epoch_s'][-1] + 1)
        save_data['loss_s'].append(avgloss)
        save_data['last_weights'] = net.state_dict()
        if save_data['best_loss'] > save_data['loss_s'][-1]: # 保存最优权重
            save_data['best_loss'] = save_data['loss_s'][-1]
            save_data['best_epoch'] = save_data['epoch_s'][-1]
            save_data['best_weights'] = net.state_dict()
        torch.save(save_data, weights_path)  # 更新权重日志文件
        print(f"save success")
