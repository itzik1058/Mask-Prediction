import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from pathlib import Path
from PIL import Image


class MaskDataset(data.Dataset):
    def __init__(self, path: Path):
        self.items = []
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        for img in path.iterdir():
            if not img.is_file():
                continue
            img_id, bbox, mask = img.stem.split('__')
            image = transform(Image.open(img))
            bbox = torch.tensor(json.loads(bbox), dtype=torch.long)
            mask = mask == 'True'
            self.items.append((image, bbox, mask))

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)
