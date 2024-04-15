import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os

def sort_by_number(name):
    number = int(name.split('.')[0])
    return number



class Datasat(Dataset):
    def __init__(self, mode):

        super(Datasat, self).__init__()
        # self.size = int(size)
        self.img_path1 = []
        self.img_path2 = []


        if mode == 'train':
            self.PAN_path = './train/MSI/'
            self.img_path1 = os.listdir(self.PAN_path)
            self.img_path1 = sorted(self.img_path1, key=sort_by_number)
            self.HS_path = './train/HSI/'
            self.img_path2 = os.listdir(self.HS_path)
            self.img_path2 = sorted(self.img_path2, key=sort_by_number)
            self.gtHS_path = './train/HSI/'
            self.img_path3 = os.listdir(self.gtHS_path)
            self.img_path3 = sorted(self.img_path3, key=sort_by_number)

        if mode == 'test':
            self.PAN_path = './test/MSI/'
            self.img_path1 = os.listdir(self.PAN_path)
            self.img_path1 = sorted(self.img_path1, key=sort_by_number)
            self.HS_path = './test/HSI/'
            self.img_path2 = os.listdir(self.HS_path)
            self.img_path2 = sorted(self.img_path2, key=sort_by_number)
            self.gtHS_path = './test/HSI/'
            self.img_path3 = os.listdir(self.gtHS_path)
            self.img_path3 = sorted(self.img_path3, key=sort_by_number)


    def __getitem__(self, item):

        self.real_PAN_path = os.path.join(self.PAN_path, self.img_path1[item])
        # data1 = loadmat(self.real_PAN_path)['pan'].reshape(154, self.size, self.size)
        data1 = loadmat(self.real_PAN_path)['hrMS_crop']
        data1 = data1.astype(np.float32)
        data1 = torch.from_numpy(data1)
        data1 = torch.squeeze(data1)
        # data1 = np.transpose(data1, (2, 0, 1))

        self.real_hs_path = os.path.join(self.HS_path, self.img_path2[item])
        # data2 = loadmat(self.real_hs_path)['hs'].reshape(154, self.size, self.size)
        data2 = loadmat(self.real_hs_path)['lrHS_crop']
        data2 = data2.astype(np.float32)
        data2 = torch.from_numpy(data2)
        data2 = torch.unsqueeze(data2,dim=0)
        data2 = torch.nn.functional.interpolate(data2, scale_factor=0.25)
        data2 = torch.squeeze(data2)
        # data2 = np.transpose(data2, (2, 0, 1))

        self.gt_hs_path = os.path.join(self.gtHS_path, self.img_path3[item])
        # data2 = loadmat(self.real_hs_path)['hs'].reshape(154, self.size, self.size)
        data3 = loadmat(self.gt_hs_path)['lrHS_crop']
        data3 = data3.astype(np.float32)
        data3 = torch.from_numpy(data3)

        data3 = torch.squeeze(data3)


        return data1, data2, data3

    def __len__(self):
        return len(self.img_path1)



if __name__ == '__main__':
    HRData = Datasat('pavia/train/hrMS')
    print(len(HRData))
    LRHSData = Datasat('pavia/train/LRHS')
    print(len(LRHSData))
