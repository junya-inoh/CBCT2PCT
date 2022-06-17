from torchvision import transforms
from torch.utils import data

import numpy as np
import pydicom

from utils import HUrescale


class ImageTransform():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ])
        }

    def __call__(self, img, phase='train'):
        # img = HUrescale(img).to256(-1000, 1200, type=np.uint8)
        img = HUrescale(img).img2var(-1000, 2200)
        return self.data_transform[phase](img)


class DCMDataset(data.Dataset):
    def __init__(self, file_list_A, file_list_B, transform='None', phase='train'):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return max(len(self.file_list_A), len(self.file_list_B))

    def __getitem__(self, index):
        path_A = self.file_list_A[index % len(self.file_list_A)]
        path_B = self.file_list_B[index % len(self.file_list_B)]

        ds_A = pydicom.dcmread(path_A, force=True)
        img_A = ds_A.pixel_array.astype(np.float32) * ds_A.RescaleSlope + ds_A.RescaleIntercept
        img_A_transformed = self.transform(img_A, phase=self.phase)

        ds_B = pydicom.dcmread(path_B, force=True)
        img_B = ds_B.pixel_array.astype(np.float32) * ds_B.RescaleSlope + ds_B.RescaleIntercept
        img_B_transformed = self.transform(img_B, phase=self.phase)

        return {'Atf': img_A_transformed, 'Btf': img_B_transformed}
        # return {'A': img_A, 'B': img_B, 'Atf': img_A_transformed, 'Btf': img_B_transformed}
