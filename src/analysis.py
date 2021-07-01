import dataset
import masknet
from train import intersection_over_union, collate
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == '__main__':
    img_data = []
    for path in [Path('../data/train'), Path('../data/test')]:
        bbox_cx, bbox_cy, bbox_w, bbox_h, proper = [], [], [], [], []
        for img in path.iterdir():
            if not img.is_file():
                continue
            img_id, bbox, mask = img.stem.split('__')
            w, h = Image.open(img).size
            bbox = json.loads(bbox)
            mask = mask == 'True'
            bbox_cx.append((bbox[0] + bbox[2] / 2) / w)
            bbox_cy.append((bbox[1] + bbox[3] / 2) / h)
            bbox_w.append(bbox[2] / w)
            bbox_h.append(bbox[3] / h)
            proper.append(mask)
        img_data.append([bbox_cx, bbox_cy, bbox_w, bbox_h, proper])
    plt.subplot(2, 1, 1)
    sns.boxplot(data=img_data[0][:4]).set(xticklabels=['Center X', 'Center Y', 'Width', 'Height'])
    plt.title('Bounding Box Coordinates (Train)')
    plt.subplot(2, 1, 2)
    sns.boxplot(data=img_data[0][:4]).set(xticklabels=['Center X', 'Center Y', 'Width', 'Height'])
    plt.title('Bounding Box Coordinates (Test)')
    plt.show()
    plt.subplot(2, 1, 1)
    plt.scatter(img_data[0][0], img_data[0][1])
    plt.title('Bounding Box Center (Train)')
    plt.subplot(2, 1, 2)
    plt.scatter(img_data[1][0], img_data[1][1])
    plt.title('Bounding Box Center (Test)')
    plt.show()
    plt.subplot(2, 1, 1)
    plt.hist2d(x=img_data[0][0], y=img_data[0][1], bins=20)
    plt.title('Bounding Box Center Distribution (Train)')
    plt.subplot(2, 1, 2)
    plt.hist2d(x=img_data[1][0], y=img_data[1][1], bins=20)
    plt.title('Bounding Box Center Distribution (Test)')
    plt.show()
    plt.subplot(1, 2, 1)
    plt.pie([sum(img_data[0][4]), len(img_data[0][4]) - sum(img_data[0][4])], labels=['True', 'False'],
            autopct=lambda pct: f'{pct:.1f}% ({int(round(len(img_data[0][4]) * pct / 100))})')
    plt.title('Proper Mask (Train)')
    plt.subplot(1, 2, 2)
    plt.pie([sum(img_data[1][4]), len(img_data[1][4]) - sum(img_data[1][4])], labels=['True', 'False'],
            autopct=lambda pct: f'{pct:.1f}% ({int(round(len(img_data[1][4]) * pct / 100))})')
    plt.title('Proper Mask (Test)')
    plt.show()
    mask_net = masknet.MaskNet().cuda()
    mask_net.load_state_dict(torch.load('masknet.torch'))
    mask_net.eval()
    test_dataset = dataset.MaskDataset(Path('../data/test'))
    test_loader = data.DataLoader(test_dataset, batch_size=32, collate_fn=collate)
    iou, accuracy = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for image, true_bbox, true_mask in test_loader:
            image, true_bbox, true_mask = image.cuda(), true_bbox.float().cuda(), true_mask.cuda()
            bbox, mask = mask_net(image)
            iou += intersection_over_union(bbox, true_bbox).sum().item() / len(test_loader.dataset)
            mask: torch.Tensor
            accuracy += mask.ge(0.5).eq(true_mask).float().sum().item() / len(test_loader.dataset)
            tp += mask.ge(0.5).eq(1).mul(true_mask.eq(1)).int().sum().item() / len(test_loader.dataset)
            fp += mask.ge(0.5).eq(1).mul(true_mask.eq(0)).int().sum().item() / len(test_loader.dataset)
            tn += mask.ge(0.5).eq(0).mul(true_mask.eq(0)).int().sum().item() / len(test_loader.dataset)
            fn += mask.ge(0.5).eq(0).mul(true_mask.eq(1)).int().sum().item() / len(test_loader.dataset)
    print(pd.DataFrame(np.array([[tp, fp], [tn, fn]]), index=['Positive', 'Negative'], columns=['True', 'False']))
    sns.heatmap(pd.DataFrame(np.array([[tp, fp], [tn, fn]]), index=['Positive', 'Negative'], columns=['True', 'False']),
                cmap="YlGnBu", annot=True)
    plt.show()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5244, 0.4904, 0.4781], std=[0.2655, 0.2623, 0.2576])
    ])
    c = 0
    for img in Path('../data/test').iterdir():
        if not img.is_file():
            continue
        c += 1
        if c > 10:
            break
        img_id, true_bbox, true_mask = img.stem.split('__')
        original = Image.open(img)
        true_bbox = json.loads(true_bbox)
        true_mask = true_mask == 'True'
        image = transform(original).cuda()
        bbox, mask = mask_net(image.unsqueeze(0))
        bbox = bbox.squeeze()
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(original)
        rect = patches.Rectangle((bbox[0].item(), bbox[1].item()), bbox[2].item(), bbox[3].item(),
                                 facecolor='none', edgecolor='r')
        ax.add_patch(rect)
        true_rect = patches.Rectangle((true_bbox[0], true_bbox[1]), true_bbox[2], true_bbox[3],
                                      facecolor='none', edgecolor='g')
        ax.add_patch(true_rect)
        title = ''
        if mask.ge(0.5).eq(1).item() and true_mask:
            title = 'True Positive'
        elif mask.ge(0.5).eq(1).item() and not true_mask:
            title = 'False Positive'
        elif mask.ge(0.5).eq(0).item() and not true_mask:
            title = 'True Negative'
        elif mask.ge(0.5).eq(0).item() and true_mask:
            title = 'False Negative'
        iou = intersection_over_union(bbox.round().long().view(1, 4), torch.tensor(true_bbox).cuda().view(1, 4)).item()
        title += f' (IoU {iou:.2f})'
        plt.title(title)
        # plt.title(f'Proper={true_mask}')
        plt.show()
