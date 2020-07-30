from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
import data_loader



class textCNN_data(Dataset):
    def __init__(self):
        trainData = data_loader.txt2ind()
        # trainData = list(filter(None, trainData))
        random.shuffle(trainData)
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        #根据索引，返回类别和句子
        data = self.trainData[idx]
        # data = list(filter(None, data))
        data = [int(x) for x in data]
        cla = data[0]
        sentence = np.array(data[1:])

        return cla, sentence


if __name__ == "__main__":
    dataset = textCNN_data()
    cla, sen = dataset.__getitem__(0)

    print(cla)
    print(sen)