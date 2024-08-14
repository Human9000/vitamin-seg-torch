import math
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import signal, ndimage
from torch.utils.data import Dataset
import wfdb


class MyDataset(Dataset):
    def __init__(self, t=10, transform=False):
        super().__init__()
        try:
            print("loading data ...")
            with open(r"D:\Code\hao\2024\ecg_img2signal\dataset\signals5000.pkl", "rb") as f:
            # with open(r"D:\Code\hao\2024\ecg_rebuild\dataset\signals_2.pkl", "rb") as f:
                # with open(r"dataset\signals.pkl", "rb") as f:
                signals = pickle.load(f)
        except:
            print("loading data error, make new data")
            signals = self.pre_load_data()
            with open("dataset/signals.pkl", "wb") as f:
                pickle.dump(signals, f)

        # with open("dataset/signals5000.pkl", "wb") as f:
        #     pickle.dump(signals[:5000], f)
        self.t = t
        print("data filter ...")
        # 筛选有效的信号
        self.t_signals = [i for i in signals if i[0].shape[1] >= t * 500]
        self.transform = transform
        # np.savez("signals.npz", signals=self.signals)

    def pre_load_data(self):
        # 读取所有的信号数据集
        signals = []
        root = r"D:\Dataset\2023\ecg\chanllenge2021lead12flatten"

        paths = []
        for fname in os.listdir(root):
            if fname.endswith("hea") and fname[0] != 'I':
                paths.append(f"{root}/{fname[:-4]}")
        # paths = paths[:30]

        for path in tqdm.tqdm(paths):
            data, fields = wfdb.rdsamp(path)
            data = data.T
            if data.shape[-1] > 10000:  # 记录长度大于10000的后面的部分裁剪掉
                data = data[..., :10000]
            # 去50hz工频噪声
            data = signal.lfilter(*signal.butter(3, 50 / 500), data)

            # 去除基线飘逸
            m = signal.medfilt2d(data, (1, 301))  # 中值滤波
            e = ndimage.uniform_filter(m, (1, 301))  # 均值滤波
            data = data - e
            signals.append((data, fields))
        return signals

    def signal_random_sample(self, signal):
        # 默认采样率为500hz  默认比例尺为 0.1mv/像素y, 10ms/像素x
        t = 10
        l = int(t * 500)
        s = random.randint(0, signal.shape[-1] - l)
        signal = signal[:, s:s + l]
        return signal

    def plot_mask_one_lead(self, lead, mask, x0, x1, y0):
        w = x1 - x0
        r = 50  # signal值和像素的比例为1:80
        signal_interp = np.interp(np.linspace(0, len(lead) - 1, w),
                                  np.linspace(0, len(lead) - 1, len(lead)),
                                  lead)
        x = np.linspace(x0, x1, w).astype("int")
        y = (signal_interp * r).astype("int") + y0
        # 画图
        xy = np.array([x, y]).T
        for p1, p2 in zip(xy[:-1], xy[1:]):  # 画矩形图
            wl, wr = min(p1[0], p2[0]), max(p1[0], p2[0])
            hl, hr = min(p1[1], p2[1]), max(p1[1], p2[1])
            mask[hl:hr + 1, wl:wr + 1] = 1

        return mask

    def plot_feature_one_lead(self, lead, feature,  x0, x1, y0):
        w = x1 - x0
        r = 50  # signal值和像素的比例为1:80
        d = 1*r
        signal_interp = np.interp(np.linspace(0, len(lead) - 1, w),
                                  np.linspace(0, len(lead) - 1, len(lead)),
                                  lead)
        y = (signal_interp * r).astype("int") + y0 # w
        hs = np.linspace(0, 1024 - 1, 1024).astype("int")
        f = np.abs(y[None] - hs[:,None])
        f = (np.clip(f, 0, d,)/d) ** 0.3
        f[f==0] = -0.01
        # img  = (f==0).astype('uint8') * 255
        # cv2.imshow("img", img)
        # cv2.waitKey()
        #
        # exit(0)
        # f = (np.clip(f, 0, d,)/d) ** 0.25
        feature[...,x0:x1] = f
        return feature

    def plot_feature(self, signal, feature):
        ys = [300, 400, 500, 600, 700, 800, 900]
        feature[-1] = self.plot_feature_one_lead(signal[1,],feature[-1],  128, 1408, ys[-1])
        for i in range(6):
            feature[i] = self.plot_feature_one_lead(signal[i, ],feature[i], 128, 1408, ys[i])
        for i in range(6):
            feature[i+6] = self.plot_feature_one_lead(signal[i + 6, ],feature[i+6],  128, 1408, ys[i])
        return feature


    def add_noise(self, mask):
        r = np.random.random(mask.shape) < 0.95
        mask = r * mask + (1 - r) * (1 - mask)
        return mask

    def random_mask(self, mask):
        w, h = mask.shape
        _m = np.random.random([w // 10 + 1, h // 5 + 1]) < 0.90
        _m = cv2.resize(_m.astype('uint8') * 255, [h, w]) / 255 > 0.8
        return mask * _m

    def plot_mask(self, signal, mask):
        ys = [300, 400, 500, 600, 700, 800, 900]
        l2 = signal.shape[-1] // 2
        mask = self.plot_mask_one_lead(signal[1,], mask, 128, 1408, ys[-1])
        for i in range(6):
            mask = self.plot_mask_one_lead(signal[i, :l2], mask, 128, 768, ys[i])
        for i in range(6):
            mask[ys[i] - 20:ys[i] + 20, 768:768 + 1, ] = 1
            mask = self.plot_mask_one_lead(signal[i + 6, l2:], mask, 768, 1408, ys[i])
        return mask

    def __len__(self):
        return len(self.t_signals)

    def __getitem__(self, item):
        signal, info = self.t_signals[item]
        signal = self.signal_random_sample(signal)
        mask = np.zeros([1024, 1536])
        mask = self.plot_mask(signal, mask)
        feature = np.ones([13, 1024, 1536], dtype=np.float32)
        feature = self.plot_feature(signal, feature)
        return mask[None], feature


if __name__ == '__main__':
    dataset = MyDataset()
    mask,feature = dataset.__getitem__(0)

    import cv2
    img = np.zeros([1024, 1536, 3])
    cols = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],

        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],

        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [1, 1, 1],
    ]
    cols = np.array(cols)
    for i in range(13):
        img = (1-feature[i][...,None]) * cols[i][None,None] + img
    cv2.imwrite(f'feature.png', img*255)
    cv2.imwrite(f'mask.png', mask[0]*255)
    print(mask.shape)

