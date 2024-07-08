import os
import copy
from pathlib import Path

import cv2
import json
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, img_dir: str, metafile: str):
        super(MyDataset, self).__init__()
        self.img_dir = img_dir
        self.cln_path = metafile
        self.df = pd.read_csv(metafile)
        self.dirs = ['acute_arms', 'control_arms', 'disease_arms']

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize([256, 256]),
            transforms.CenterCrop([224, 224]),
            transforms.Normalize(mean=[0.3867] * 3, std=[0.2395] * 3)
        ])

        self.images, self.labels = self.read_data

    def __len__(self):
        return len(self.df)

    @property
    def read_data(self):
        images, labels = [], []

        for record in self.df.itertuples():
            name, label = record.No, record.label - 1
            category = self.dirs[label]
            img_path = os.path.join(self.img_dir, category, name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            images.append(img)
            labels.append(label)

        return images, labels


    def __getitem__(self, idx):
        img = self.img_transform(self.images[idx])
        label = torch.as_tensor(self.labels[idx])
        return img, label


class NewDataset(Dataset):
    def __init__(self, img_dir: str, metafile: str):
        super(NewDataset, self).__init__()
        self.img_dir = img_dir
        self.cln_path = metafile
        self.df = pd.read_csv(metafile)
        self.dirs = ['acute_arms', 'control_arms', 'disease_arms']
        self.mean = torch.tensor([1.439516129032258, 5.869086021505376, 20.526881720430108, 185.06720430107526, 155.1478494623656, 97.48655913978494, 24.31451612903226, 104.50268817204301, 166.85228494623655, 101.60485215053762, 87.17002688172043, 43.94529569892473, 104.36520161290323, 52.962096774193554, 1.7073655913978492, 20.12598252688172, 147.8970564516129, 48.34153225806451, 82.9502688172043, 79.96626344086023, 67.33333333333333, 58.54495967741935, 114.086438172043, 109.52353494623657, 65.53810483870969, 334.43817204301075, 738.5873655913979, 652.7822580645161, 616.8763440860215, 1116.0067204301076, 605.4663978494624, 298.61155913978496, 778.1008064516129, 660.2261290322581, 20.01309139784946, 4.189284946236559, 108.79623655913977, 33.02150537634409, 81.79758064516128, 27.42288978494624, 327.4685483870968, 212.5619623655914, 41.939516129032256, 14.627715053763442, 9.75228494623656, 8.702768817204301, 22.238803763440856, 0.20372311827956988, 0.056115591397849475, 0.3137096774193549, 28.384596774193547, 47.54278225806452, 7.884327956989247, 1.5591666666666666, 0.4676881720430108, 0.20274193548387098, 0.8952355316285331])
        self.var = torch.tensor([0.4966621099116323, 3.958412507653351, 24.848706739031908, 114.80334077107875, 129.34654973393864, 126.02847349447502, 75.25506563281664, 131.71513776448373, 55.49243247566955, 11.925462460241468, 10.534299652956554, 34.76292411533258, 82.58508202316091, 41.92270094849583, 2.2933486939945533, 47.26685481933908, 19.07300134803412, 11.393688286805348, 9.608283490661943, 9.921144216966272, 10.925336558399984, 6.559961407864024, 12.461225636439424, 18.31085704909147, 12.263464445210968, 82.10939716937469, 359.3874411668006, 154.64327346075353, 170.08538993791024, 296.9504877055835, 126.96255401124448, 80.52991335903859, 237.65800280684294, 156.32888242425201, 47.358203109238794, 4.381215409885884, 28.025100974057974, 8.237812557886198, 10.28037734414631, 11.689809730283553, 31.995175949309388, 142.57941166923348, 14.358266231924356, 7.4926724829124245, 4.647860347335432, 3.7653382854780717, 12.048930001434766, 0.18707778649450793, 0.5645404210722315, 1.9350907627675906, 20.56667531104567, 27.752970646973928, 9.140700167260169, 2.7328163444921048, 0.43394745676179763, 1.219520903243442, 2.523228671687364])

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize([256, 256]),
            transforms.CenterCrop([224, 224]),
            transforms.Normalize(mean=[0.3867] * 3, std=[0.2395] * 3)
        ])

        self.images, self.cln_datas, self.labels = self.read_data

    def __len__(self):
        return len(self.df)

    @property
    def read_data(self):
        images, labels, cln_datas = [], [], []

        for record in self.df.itertuples():
            name, label = record.No, record.label-1
            category = self.dirs[label]
            img_path = os.path.join(self.img_dir, category, f"{name}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            images.append(img)
            cln_datas.append(record[3:])
            labels.append(label)

        return images, cln_datas, labels

    def __getitem__(self, idx):
        img = self.img_transform(self.images[idx])
        cln_data = (torch.tensor(self.cln_datas[idx]) - self.mean) / self.var
        label = torch.as_tensor(self.labels[idx])
        return img, cln_data, label


class ClnDataset(Dataset):
    def __init__(self, img_dir: str, metafile: str):
        super(ClnDataset, self).__init__()
        self.img_dir = img_dir
        self.cln_path = metafile
        self.df = pd.read_csv(metafile)
        self.dirs = ['acute_arms', 'control_arms', 'disease_arms']
        self.mean = torch.tensor([1.439516129032258, 5.869086021505376, 20.526881720430108, 185.06720430107526, 155.1478494623656, 97.48655913978494, 24.31451612903226, 104.50268817204301, 166.85228494623655, 101.60485215053762, 87.17002688172043, 43.94529569892473, 104.36520161290323, 52.962096774193554, 1.7073655913978492, 20.12598252688172, 147.8970564516129, 48.34153225806451, 82.9502688172043, 79.96626344086023, 67.33333333333333, 58.54495967741935, 114.086438172043, 109.52353494623657, 65.53810483870969, 334.43817204301075, 738.5873655913979, 652.7822580645161, 616.8763440860215, 1116.0067204301076, 605.4663978494624, 298.61155913978496, 778.1008064516129, 660.2261290322581, 20.01309139784946, 4.189284946236559, 108.79623655913977, 33.02150537634409, 81.79758064516128, 27.42288978494624, 327.4685483870968, 212.5619623655914, 41.939516129032256, 14.627715053763442, 9.75228494623656, 8.702768817204301, 22.238803763440856, 0.20372311827956988, 0.056115591397849475, 0.3137096774193549, 28.384596774193547, 47.54278225806452, 7.884327956989247, 1.5591666666666666, 0.4676881720430108, 0.20274193548387098, 0.8952355316285331])
        self.var = torch.tensor([0.4966621099116323, 3.958412507653351, 24.848706739031908, 114.80334077107875, 129.34654973393864, 126.02847349447502, 75.25506563281664, 131.71513776448373, 55.49243247566955, 11.925462460241468, 10.534299652956554, 34.76292411533258, 82.58508202316091, 41.92270094849583, 2.2933486939945533, 47.26685481933908, 19.07300134803412, 11.393688286805348, 9.608283490661943, 9.921144216966272, 10.925336558399984, 6.559961407864024, 12.461225636439424, 18.31085704909147, 12.263464445210968, 82.10939716937469, 359.3874411668006, 154.64327346075353, 170.08538993791024, 296.9504877055835, 126.96255401124448, 80.52991335903859, 237.65800280684294, 156.32888242425201, 47.358203109238794, 4.381215409885884, 28.025100974057974, 8.237812557886198, 10.28037734414631, 11.689809730283553, 31.995175949309388, 142.57941166923348, 14.358266231924356, 7.4926724829124245, 4.647860347335432, 3.7653382854780717, 12.048930001434766, 0.18707778649450793, 0.5645404210722315, 1.9350907627675906, 20.56667531104567, 27.752970646973928, 9.140700167260169, 2.7328163444921048, 0.43394745676179763, 1.219520903243442, 2.523228671687364])

        self.cln_datas, self.labels = self.read_data

    def __len__(self):
        return len(self.df)

    @property
    def read_data(self):
        """读取所有图片"""
        images, labels, cln_datas = [], [], []

        for record in self.df.itertuples():
            name, label = record.No, record.label-1
            cln_datas.append(record[3:])
            labels.append(label)

        return cln_datas, labels

    def __getitem__(self, idx):
        cln_data = (torch.tensor(self.cln_datas[idx]) - self.mean) / self.var
        label = torch.as_tensor(self.labels[idx])
        return cln_data, label


class TotalDataset(Dataset):
    def __init__(self, path, patient_no: list, is_train=True) -> None:
        super().__init__()
        self.data = {}
        self.patient_no = patient_no
        self.path = path
        self.mean = torch.tensor([1.439516129032258, 5.869086021505376, 20.526881720430108, 185.06720430107526, 155.1478494623656, 97.48655913978494, 24.31451612903226, 104.50268817204301, 166.85228494623655, 101.60485215053762, 87.17002688172043, 43.94529569892473, 104.36520161290323, 52.962096774193554, 1.7073655913978492, 20.12598252688172, 147.8970564516129, 48.34153225806451, 82.9502688172043, 79.96626344086023, 67.33333333333333, 58.54495967741935, 114.086438172043, 109.52353494623657, 65.53810483870969, 334.43817204301075, 738.5873655913979, 652.7822580645161, 616.8763440860215, 1116.0067204301076, 605.4663978494624, 298.61155913978496, 778.1008064516129, 660.2261290322581, 20.01309139784946, 4.189284946236559, 108.79623655913977, 33.02150537634409, 81.79758064516128, 27.42288978494624, 327.4685483870968, 212.5619623655914, 41.939516129032256, 14.627715053763442, 9.75228494623656, 8.702768817204301, 22.238803763440856, 0.20372311827956988, 0.056115591397849475, 0.3137096774193549, 28.384596774193547, 47.54278225806452, 7.884327956989247, 1.5591666666666666, 0.4676881720430108, 0.20274193548387098, 0.8952355316285331])
        self.var = torch.tensor([0.4966621099116323, 3.958412507653351, 24.848706739031908, 114.80334077107875, 129.34654973393864, 126.02847349447502, 75.25506563281664, 131.71513776448373, 55.49243247566955, 11.925462460241468, 10.534299652956554, 34.76292411533258, 82.58508202316091, 41.92270094849583, 2.2933486939945533, 47.26685481933908, 19.07300134803412, 11.393688286805348, 9.608283490661943, 9.921144216966272, 10.925336558399984, 6.559961407864024, 12.461225636439424, 18.31085704909147, 12.263464445210968, 82.10939716937469, 359.3874411668006, 154.64327346075353, 170.08538993791024, 296.9504877055835, 126.96255401124448, 80.52991335903859, 237.65800280684294, 156.32888242425201, 47.358203109238794, 4.381215409885884, 28.025100974057974, 8.237812557886198, 10.28037734414631, 11.689809730283553, 31.995175949309388, 142.57941166923348, 14.358266231924356, 7.4926724829124245, 4.647860347335432, 3.7653382854780717, 12.048930001434766, 0.18707778649450793, 0.5645404210722315, 1.9350907627675906, 20.56667531104567, 27.752970646973928, 9.140700167260169, 2.7328163444921048, 0.43394745676179763, 1.219520903243442, 2.523228671687364])

        if is_train:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([240, 240]),
                transforms.RandomCrop([224, 224]),
                transforms.Normalize(mean=[0.3867] * 3, std=[0.2395] * 3)
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                transforms.Normalize(mean=[0.3867] * 3, std=[0.2395] * 3)
            ])

        self._get_data()

    def __len__(self):
        return len(self.patient_no)

    @staticmethod
    def _get_img(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_data(self):
        path = Path(self.path)

        for idx, no in enumerate(self.patient_no):
            json_path = path / f"{no}.json"
            with open(json_path, 'r') as f:
                csv_data = json.load(f)
            img_path = path / csv_data["图片路径"]
            img_path = str(img_path)

            csv_data["image"] = self._get_img(img_path)
            del csv_data["图片路径"]
            # endregion
            self.data[idx] = csv_data

    def __getitem__(self, item):
        data_item = copy.deepcopy(self.data[item])
        data_item['image'] = self.img_transform(data_item['image'])
        return data_item


    



if __name__ == '__main__':
    # dataset = NewDataset(r"/data_net/lhz/wangqiuming/dataset/Heterolymph", r"/data_net/lhz/wangqiuming/dataset/Heterolymph/csv_data_more/test_fold1.csv")
    train_no = np.load("/data_net/lhz/wangqiuming/dataset/Heterolymph/fold/train_0.npy")
    dataset = TotalDataset(r"/data_net/lhz/wangqiuming/dataset/Heterolymph/csv_data7_21", train_no)

    print(len(dataset))

    da = dataset[1]


    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8)
    for i, data in enumerate(dataloader):
        print("round {} -----".format(i))
        print(data['image'].shape)
        merged_tensor = torch.stack(list(data['血常规报告参数'].values()), dim=1)
        print(merged_tensor.shape)
        print(data['基础信息']['序号'])
        print(data.keys())

