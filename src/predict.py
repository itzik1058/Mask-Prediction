import os
import argparse
import pandas as pd
import masknet
import torch
import torchvision.transforms as transforms
from PIL import Image


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.5244, 0.4904, 0.4781], std=[0.2655, 0.2623, 0.2576])
])
mask_net = masknet.MaskNet().cuda()
mask_net.load_state_dict(torch.load('masknet.torch'))
mask_net.eval()
predictions = []
# mask_accuracy = 0
with torch.no_grad():
    for filename in files:
        # img_id, true_bbox, true_mask = filename.replace('.jpg', '').split('__')
        # true_mask = true_mask == 'True'
        # print(filename)
        image = transform(Image.open(os.path.join(args.input_folder, filename))).cuda()
        bbox, mask = mask_net(image.unsqueeze(0))
        bbox = bbox.squeeze().cpu().numpy()
        predictions.append([filename, bbox[3], bbox[2], bbox[1], bbox[0], mask.item() > 0.5])
        # mask_accuracy += mask.ge(0.5).item() == true_mask
# mask_accuracy /= len(files)
# print(f'Accuracy {mask_accuracy:.3f}')
prediction_df = pd.DataFrame(data=predictions, columns=['filename', 'h', 'w', 'y', 'x', 'proper_mask'])
prediction_df.to_csv("prediction.csv", index=False, header=True)
