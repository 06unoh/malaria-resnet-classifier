import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, filepath, files, labels, transform):
        self.filepath = filepath
        self.files = files
        self.labels = labels
        self.tf = transform
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.filepath, self.files[idx]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.tf:
            img = self.tf(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label