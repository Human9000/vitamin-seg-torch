import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from network import SegVitamin
from dataset import EcgDataset, DataLoader


def make_feature_img(feature):
    img = np.zeros([1024, 1536, 3])
    cols = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],

        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],

        [1, 1, 1],
    ]
    cols = np.array(cols)
    for i in range(13):
        img += (1 - feature[i][..., None]) * cols[i][None, None]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype('uint8')
    return img


# feature转成signal信号
def feature_to_signal(feature):
    # feature.shape = (1, 13,1024,1536)
    # print(feature.shape)
    m = (feature < 0.5).max(dim=1)[0].max(dim=1)[0][0]  # (1536)
    index = torch.argwhere(m == 1)  # (-1)
    l, r = min(index), max(index) + 1

    f2 = feature[0,:, :, l:r] # (13, 1024, r-l)
    x = torch.argmin(f2, dim=1) # (13, r-l)
    # print(x.shape)
    # 线性插值还原信号
    x  = torch.nn.functional.interpolate(x[None,].float(), (1000,), mode='linear', align_corners=False)[0]
    return x





if __name__ == '__main__':
    net = SegVitamin(size=(1024, 1536), fact=(2, 2), in_chans=1, out_channel=13)
    net.eval()
    net.cuda()
    save_data = torch.load(f"weights/SegVitamin2.pth")
    fig = plt.figure()
    plt.plot(save_data['loss_s'][2:])
    plt.savefig("plots/loss.png")
    plt.close(fig)

    state_dict = save_data['last_weights']
    net.load_state_dict(state_dict, strict=False)

    ecg_dataset = EcgDataset(10, True)  # 获取 10s 的数据
    loader = DataLoader(ecg_dataset, 1, True, num_workers=0)
    loader.dataset.t = 10
    count = 0
    with torch.no_grad():
        with tqdm(loader) as tbr:
            for mask, feature in tbr:
                mask = mask.cuda().float()
                feature = feature.cuda().float()
                p = net(mask)
                p = torch.clip(p, 0, 1)

                f_signal = feature_to_signal(feature)
                p_signal = feature_to_signal(p)

                # print(f_signal.shape)
                # print(p_signal.shape)

                p = p.cpu().numpy()
                feature = feature.cpu().numpy()
                fig = plt.figure(figsize=(14, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(make_feature_img(p[0]))
                plt.subplot(1, 2, 2)
                plt.imshow(make_feature_img(feature[0]))
                plt.tight_layout()
                plt.savefig(f"plots/{count}.png")
                plt.close(fig)

                count += 1
                if count > 100:
                    break
