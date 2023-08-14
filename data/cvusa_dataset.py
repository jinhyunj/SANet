from data.base_dataset import BaseDataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class CVUSADataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dataset_list = np.load(opt.dataroot, allow_pickle=True)

    def __getitem__(self, index):

        A_img = Image.open(self.dataset_list[index]['aerial']).convert('RGB')
        G_img = Image.open(self.dataset_list[index]['ground']).convert('RGB')
        G_seg = Image.open(self.dataset_list[index]['semantic']).convert('RGB')

        A_transform = transforms.Compose([transforms.CenterCrop(400),
                                         transforms.Resize([256,256]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])

        G_transform = transforms.Compose([transforms.Resize([128,512]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])

        A = A_transform(A_img)
        G = G_transform(G_img)
        G_seg = G_transform(G_seg)

        A_path = self.dataset_list[index]['aerial']
        G_path = self.dataset_list[index]['ground']

        return {'A': A, 'G': G, 'G_seg': G_seg, 'A_paths': A_path, 'G_paths': G_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset_list)
