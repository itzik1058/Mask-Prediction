import dataset
import masknet
import torch
import torch.utils.data as data
import time
from pathlib import Path
import matplotlib.pyplot as plt


def collate(batch):
    images, bounding_boxes, masks = [], [], []
    for image, bbox, mask in batch:
        images.append(image)
        bounding_boxes.append(bbox)
        masks.append(mask)
    return torch.stack(images), torch.stack(bounding_boxes), torch.tensor(masks, dtype=torch.long)


def intersection_over_union(bbox1, bbox2):
    tl1, tl2 = bbox1[:, :2], bbox2[:, :2]
    br1, br2 = tl1 + bbox1[:, 2:], tl2 + bbox2[:, 2:]
    area1 = bbox1[:, 2] * bbox1[:, 3]
    area2 = bbox2[:, 2] * bbox2[:, 3]
    min_b = torch.max(tl1, tl2)
    max_b = torch.min(br1, br2)
    overlap = torch.clamp(max_b - min_b, min=0)
    intersection = overlap[:, 0] * overlap[:, 1]
    union = area1 + area2 - intersection
    return intersection / union


def train(train_path: Path, test_path: Path, n_epoch=60, batch_size=32, lr=0.02):
    mask_dataset = dataset.MaskDataset(train_path)
    test_dataset = dataset.MaskDataset(test_path)
    data_loader = data.DataLoader(mask_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    model = masknet.MaskNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCELoss()
    bbox_loss, mask_loss = [], []
    bbox_iou, mask_accuracy = [], []
    bbox_test_iou, mask_test_accuracy = [], []
    for epoch in range(n_epoch):
        epoch_start = time.time()
        bbox_epoch_loss, mask_epoch_loss = 0, 0
        bbox_epoch_iou, mask_epoch_accuracy = 0, 0
        for image, true_bbox, true_mask in data_loader:
            image, true_bbox, true_mask = image.cuda(), true_bbox.float().cuda(), true_mask.cuda()
            bbox, mask = model(image)
            bbox_batch_loss = l1_loss(bbox, true_bbox)
            mask_batch_loss = bce_loss(mask, true_mask.float())
            loss = bbox_batch_loss + mask_batch_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bbox_epoch_loss += bbox_batch_loss.item() / len(data_loader)
            mask_epoch_loss += mask_batch_loss.item() / len(data_loader)
            bbox_epoch_iou += intersection_over_union(bbox, true_bbox).sum().item() / len(data_loader.dataset)
            mask_epoch_accuracy += mask.ge(0.5).eq(true_mask).float().sum().item() / len(data_loader.dataset)
        bbox_loss.append(bbox_epoch_loss)
        mask_loss.append(mask_epoch_loss)
        bbox_iou.append(bbox_epoch_iou)
        mask_accuracy.append(mask_epoch_accuracy)
        print(f'Epoch {epoch + 1}/{n_epoch} done in {time.time() - epoch_start:.2f}s')
        print(f'\tBounding Box Loss {bbox_epoch_loss:.3f}\tMask Loss {mask_epoch_loss:.3f}')
        print(f'\tBounding Box IoU {bbox_epoch_iou:.3f}\tMask Accuracy {mask_epoch_accuracy:.3f}')
        bbox_epoch_iou, mask_epoch_accuracy = 0, 0
        with torch.no_grad():
            for image, true_bbox, true_mask in test_loader:
                image, true_bbox, true_mask = image.cuda(), true_bbox.float().cuda(), true_mask.cuda()
                bbox, mask = model(image)
                bbox_epoch_iou += intersection_over_union(bbox, true_bbox).sum().item() / len(test_loader.dataset)
                mask: torch.Tensor
                mask_epoch_accuracy += mask.ge(0.5).eq(true_mask).float().sum().item() / len(test_loader.dataset)
            bbox_test_iou.append(bbox_epoch_iou)
            mask_test_accuracy.append(mask_epoch_accuracy)
        print(f'\tBounding Box IoU {bbox_epoch_iou:.3f}\tMask Accuracy {mask_epoch_accuracy:.3f}\t(Test)')
    torch.save(model.state_dict(), 'masknet.torch')
    plots = [(bbox_loss, 'Bounding Box Loss'), (mask_loss, 'Mask Loss'),
             (bbox_iou, 'Bounding Box IoU'), (mask_accuracy, 'Mask Accuracy'),
             (bbox_test_iou, 'Bounding Box IoU (Test)'), (mask_test_accuracy, 'Mask Accuracy (Test)')]
    fig = plt.figure()
    fig.suptitle('Evaluation')
    for idx, (line, title) in enumerate(plots):
        plt.subplot(3, 2, idx + 1)
        plt.plot(line)
        plt.xlabel('Epoch')
        plt.title(title)
    plt.subplots_adjust()
    plt.show()


if __name__ == '__main__':
    train(Path('../data/train'), Path('../data/test'))
