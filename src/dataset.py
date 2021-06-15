import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from pathlib import Path
from PIL import Image


class MaskDataset(data.Dataset):
    def __init__(self, path: Path):
        self.items = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5244, 0.4904, 0.4781], std=[0.2655, 0.2623, 0.2576])
        ])
        for img in path.iterdir():
            if not img.is_file():
                continue
            img_id, bbox, mask = img.stem.split('__')
            image = transform(Image.open(img))
            bbox = torch.tensor(json.loads(bbox), dtype=torch.long)
            mask = mask == 'True'
            self.items.append((image, bbox, mask))

    def img_mean_std(self):
        means = []
        stds = []
        for img, _, _ in self.items:
            means.append(torch.mean(img.view(3, -1), dim=1).unsqueeze(0))
            stds.append(torch.std(img.view(3, -1), dim=1).unsqueeze(0))
        mean = torch.mean(torch.cat(means), dim=0)  # 0.5244, 0.4904, 0.4781
        std = torch.mean(torch.cat(stds), dim=0)    # 0.2655, 0.2623, 0.2576
        return mean, std

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)
