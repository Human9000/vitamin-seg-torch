import math
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm 
from torch.utils.data import Dataset 

class MyDataset(Dataset):
    def __init__(self, transform=False):
        super().__init__()
        try:
            print("loading data ...")
            with open(r"dataset\data.pkl", "rb") as f:  
                datas = pickle.load(f)
        except:
            print("loading data error, make new data")
            datas = self.pre_load_data()
            with open("dataset/data.pkl", "wb") as f:
                pickle.dump(signals, f)
  
        self.datas = datas
        self.transform = transform 
    def pre_load_data(self): 
        datas = []
        root = r"" # ==================这里填写你的数据集根目录
        paths = []
        for fname in os.listdir(root):
            if fname.endswith("hea") and fname[0] != 'I':
                paths.append(f"{root}/{fname[:-4]}") 
        for path in tqdm.tqdm(paths): 
            # ====================== 这里添加你的数据集文件的io
            datas.append((data, fields))
        return datas 

    def __len__(self):
        return len(self.t_signals)

    def __getitem__(self, item):
        data = self.datas[item]
        # 这里可以添加transform的函数
        if self.transform:
            data = self.transform(data)
        return data

