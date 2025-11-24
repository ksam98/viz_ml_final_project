import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import argparse
import json
import glob
from tqdm import tqdm

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, transforms=None):
        self.ds = VOCDetection(root=root, year=year, image_set=image_set, download=False)
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
        
        # Get image ID from the annotation
        image_id = target['annotation']['filename']
        
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
        
        image_id_tensor = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target_dict = {}
        target_dict["boxes"] = boxes
        target_dict["labels"] = labels
        target_dict["image_id"] = image_id_tensor
        target_dict["area"] = area
        target_dict["iscrowd"] = iscrowd

        return img, target_dict, image_id

    def __len__(self):
        return len(self.ds)

def load_model(checkpoint_dir, num_classes=21, device='cpu'):
    """
    Load model from checkpoint if available, otherwise load base COCO model.
    """
    # Check for checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if checkpoint_files:
        # Sort by epoch number and get the latest
        checkpoint_files.sort()
        latest_checkpoint = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        # Load the base model architecture
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found. Loading base COCO pre-trained model...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print("Loaded COCO pre-trained model (note: class predictions may not align with VOC)")
    
    model.to(device)
    model.eval()
    return model

def evaluate(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load validation dataset
    print("Loading Pascal VOC validation dataset...")
    dataset_val = VOCDataset(
        root='./data', year='2012', image_set='val', transforms=get_transform()
    )

    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )

    # Load model
    model = load_model(args.checkpoint_dir, num_classes=21, device=device)

    # Run inference
    print(f"Running inference on validation set (max {args.max_images if args.max_images else 'all'} images)...")
    predictions = []
    
    with torch.no_grad():
        for idx, (images, targets, image_ids) in enumerate(tqdm(data_loader_val, desc="Evaluating")):
            if args.max_images and idx >= args.max_images:
                break
            
            images = list(image.to(device) for image in images)
            
            # Get predictions
            outputs = model(images)
            
            # Process each image in the batch (batch_size=1 in this case)
            for i, output in enumerate(outputs):
                image_id = image_ids[i]
                
                # Extract predictions
                boxes = output['boxes'].cpu().numpy().tolist()
                labels = output['labels'].cpu().numpy().tolist()
                scores = output['scores'].cpu().numpy().tolist()
                
                # Convert label indices to class names
                class_names = [dataset_val.classes[label] for label in labels]
                
                prediction = {
                    'image_id': image_id,
                    'predictions': []
                }
                
                # Store each detection
                for box, label, score, class_name in zip(boxes, labels, scores, class_names):
                    prediction['predictions'].append({
                        'bbox': box,  # [xmin, ymin, xmax, ymax]
                        'label': int(label),
                        'class_name': class_name,
                        'score': float(score)
                    })
                
                predictions.append(prediction)

    # Save predictions to JSON
    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"Total images evaluated: {len(predictions)}")
    
    # Print some statistics
    total_detections = sum(len(pred['predictions']) for pred in predictions)
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(predictions):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN on Pascal VOC')
    parser.add_argument('--checkpoint_dir', type=str, default='./voc_fasterrcnn_checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_file', type=str, default='./results/predictions.json',
                        help='Output JSON file for predictions')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to evaluate (default: all)')
    args = parser.parse_args()
    evaluate(args)
