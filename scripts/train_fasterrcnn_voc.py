import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # We could add more augmentations here like RandomHorizontalFlip
    # but let's keep it simple for now to ensure it works.
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, transforms=None):
        self.ds = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transforms = transforms
        
        # VOC class names
        self.classes = [
            "__background__", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        
        # Convert PIL image to tensor
        if self.transforms is not None:
            img = self.transforms(img)

        # Parse XML target into Faster R-CNN format
        boxes = []
        labels = []
        
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
            
        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target_dict = {}
        target_dict["boxes"] = boxes
        target_dict["labels"] = labels
        target_dict["image_id"] = image_id
        target_dict["area"] = area
        target_dict["iscrowd"] = iscrowd

        return img, target_dict

    def __len__(self):
        return len(self.ds)

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print("Loading Pascal VOC dataset...")
    dataset = VOCDataset(
        root='./data', year='2012', image_set='train', transforms=get_transform(train=True)
    )
    dataset_test = VOCDataset(
        root='./data', year='2012', image_set='val', transforms=get_transform(train=False)
    )

    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    data_loader_test = DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )

    # Model setup
    num_classes = 21 # 20 classes + background

    # TODO: add option to train model from scratch 
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    output_dir = "./voc_fasterrcnn_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss / len(data_loader):.4f}")

        # Save checkpoint
        checkpoint_name = f"fasterrcnn_voc_epoch_{epoch+1}.pth"
        save_path = os.path.join(output_dir, checkpoint_name)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, save_path)
        print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on Pascal VOC')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    args = parser.parse_args()
    train(args)
