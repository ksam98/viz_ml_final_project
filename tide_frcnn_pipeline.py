"""
Faster R-CNN + TIDE + Grad-CAM Pipeline (FIXED)

Requirements:
    pip install torch torchvision tidecv pytorch-grad-cam pycocotools pillow numpy tqdm opencv-python-headless

Usage:
    python tide_frcnn_pipeline.py \
        --epochs 5 \
        --train-images datasets/coco/train2017 \
        --train-annotations datasets/coco/annotations/instances_train2017.json \
        --val-images datasets/coco/val2017 \
        --val-annotations datasets/coco/annotations/instances_val2017.json \
        --output ./tide_analysis

    python tide_frcnn_pipeline.py \
        --epochs 15 \
        --train-images /scratch/as20410/Viz_project/datasets/dummy_val/images \
        --train-annotations /scratch/as20410/Viz_project/datasets/dummy_val/annotations.json \
        --val-images /scratch/as20410/Viz_project/datasets/dummy_val/images \
        --val-annotations /scratch/as20410/Viz_project/datasets/dummy_val/annotations.json \
        --output ./tide_analysis_dummy
"""

import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PIL import Image
import numpy as np
from tqdm import tqdm

import tidecv
from tidecv import TIDE, datasets as tide_datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN

# -------------------------
# Collate function
# -------------------------
def collate_fn(batch):
    images = []
    targets = []
    for img, ann in batch:
        if len(ann) == 0:
            continue

        img_tensor = TF.to_tensor(img)
        images.append(img_tensor)

        boxes = []
        labels = []
        image_id = ann[0]['image_id'] if 'image_id' in ann[0] else 0
        for obj in ann:
            x, y, w, h = obj['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(obj['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id])
        }
        targets.append(target)
    
    if len(images) == 0:
        return [], []
    
    return images, targets


import torch
from typing import List, Dict
from copy import deepcopy


def compute_iou_matrix(pred_boxes, gt_boxes):
    """Return IoU matrix [num_preds, num_gts]."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.zeros((len(pred_boxes), len(gt_boxes)))
    
    # pred: [N,4], gt: [M,4]
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]

    iou_matrix = torch.zeros((N, M))

    for i in range(N):
        for j in range(M):
            # xyxy format
            boxA = pred_boxes[i]
            boxB = gt_boxes[j]

            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            inter = max(0, xB - xA) * max(0, yB - yA)
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            union = areaA + areaB - inter
            iou = inter / union if union > 0 else 0

            iou_matrix[i, j] = iou

    return iou_matrix



def get_errors_by_image(outputs, targets, pos_thresh=0.5, bg_thresh=0.1):
    """
    TIDE-style error detection without importing external Error classes.
    Returns:
        { image_id : [error_dicts] }
    Each error_dict has:
        - 'type': str (e.g., 'Cls', 'Loc', 'Dupe', 'Bkg', 'Both', 'Miss')
        - 'pred_class' or 'gt_class'
        - 'pred_box' or 'gt_box'
    """
    from copy import deepcopy

    errors_by_image = {}

    for output, target in zip(outputs, targets):
        image_id = int(target["image_id"].item())
        preds = deepcopy(output)
        gts = deepcopy(target)

        pred_boxes = preds.get("boxes", [])
        pred_labels = preds.get("labels", [])
        pred_scores = preds.get("scores", [])

        gt_boxes = gts.get("boxes", [])
        gt_labels = gts.get("labels", [])

        gt_used = [False] * len(gt_boxes)
        errors = []

        # No predictions → all GT missed
        if len(pred_boxes) == 0:
            for i, gt_box in enumerate(gt_boxes):
                errors.append({
                    "type": "Miss",
                    "gt_class": int(gt_labels[i]),
                    "gt_box": gt_box.tolist()
                })
            errors_by_image[image_id] = errors
            continue

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        # Match predictions to GTs
        pred_matched = [None] * len(pred_boxes)
        for p_idx in range(len(pred_boxes)):
            iou_vals = iou_matrix[p_idx]
            best_gt_idx = torch.argmax(iou_vals).item()
            best_iou = float(iou_vals[best_gt_idx])
            best_gt_class = int(gt_labels[best_gt_idx])

            if best_iou >= pos_thresh and not gt_used[best_gt_idx]:
                pred_matched[p_idx] = best_gt_idx
                gt_used[best_gt_idx] = True
            else:
                pred_matched[p_idx] = None

        # Classify errors
        for p_idx in range(len(pred_boxes)):
            matched_idx = pred_matched[p_idx]
            pred_box = pred_boxes[p_idx]
            pred_cls = int(pred_labels[p_idx])

            if matched_idx is not None:
                continue  # True positive

            best_gt_idx = torch.argmax(iou_matrix[p_idx]).item()
            best_iou = float(iou_matrix[p_idx, best_gt_idx])
            best_gt_class = int(gt_labels[best_gt_idx])

            # 1. BackgroundError (no GTs)
            if len(gt_boxes) == 0:
                errors.append({
                    "type": "Bkg",
                    "pred_class": pred_cls,
                    "pred_box": pred_box.tolist()
                })
                continue

            # 2. BoxError
            if bg_thresh <= best_iou < pos_thresh and pred_cls == best_gt_class:
                errors.append({
                    "type": "Loc",
                    "pred_class": pred_cls,
                    "pred_box": pred_box.tolist(),
                    "gt_class": best_gt_class,
                    "gt_box": gt_boxes[best_gt_idx].tolist()
                })
                continue

            # 3. ClassError
            if best_iou >= pos_thresh and pred_cls != best_gt_class:
                errors.append({
                    "type": "Cls",
                    "pred_class": pred_cls,
                    "pred_box": pred_box.tolist(),
                    "gt_class": best_gt_class,
                    "gt_box": gt_boxes[best_gt_idx].tolist()
                })
                continue

            # 4. DuplicateError
            if best_iou >= pos_thresh and gt_used[best_gt_idx]:
                errors.append({
                    "type": "Dupe",
                    "pred_class": pred_cls,
                    "pred_box": pred_box.tolist(),
                    "gt_class": best_gt_class,
                    "gt_box": gt_boxes[best_gt_idx].tolist()
                })
                continue

            # 5. BackgroundError (IoU <= bg_thresh)
            if best_iou <= bg_thresh:
                errors.append({
                    "type": "Bkg",
                    "pred_class": pred_cls,
                    "pred_box": pred_box.tolist()
                })
                continue

            # 6. Fallback
            errors.append({
                "type": "Both",
                "pred_class": pred_cls,
                "pred_box": pred_box.tolist()
            })

        # Missed GTs
        for g_idx in range(len(gt_boxes)):
            if not gt_used[g_idx]:
                errors.append({
                    "type": "Miss",
                    "gt_class": int(gt_labels[g_idx]),
                    "gt_box": gt_boxes[g_idx].tolist()
                })

        if len(errors) > 0:
            errors_by_image[image_id] = errors

    return errors_by_image


from PIL import Image, ImageDraw


# -------------------------
# Grad-CAM wrapper
# -------------------------
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAMWrapperFRCNN:
    def __init__(self, model, target_layer, name, device='cuda'):
        self.model = model.eval().to(device)
        self.device = device
        self.target_layer = target_layer
        self.name = name 

    def generate_cams(self, img_tensor, gt_boxes, gt_labels):
        """Generate EigenCAM for the backbone"""
        cams = {}
        try:
            cam = AblationCAM(
                model=self.model,
                target_layers=[self.target_layer],
                use_cuda=(self.device == 'cuda'),
                reshape_transform=fasterrcnn_reshape_transform,
                ablation_layer=AblationLayerFasterRCNN(),
                ratio_channels_to_ablate = 0.1 
            )

            # img_tensor can be [C,H,W] or [1,C,H,W]
            input_tensor = img_tensor.unsqueeze(0) if img_tensor.ndim == 3 else img_tensor
            targets = [FasterRCNNBoxScoreTarget(labels=gt_labels, bounding_boxes=gt_boxes)]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)  
            cams[self.name] = grayscale_cam[0]  # first image in batch

        except Exception as e:
            print(f"Failed to generate CAM: {e}")
            cams[self.name] = None

        return cams

# -------------------------
# Analyzer
# -------------------------
class AnalyzerFRCNN:
    def __init__(self, output_dir='./tide_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gradcam_dir = self.output_dir / 'gradcams'
        self.gradcam_dir.mkdir(exist_ok=True)
        
        self.preds_dir = self.output_dir / 'predictions'
        self.preds_dir.mkdir(exist_ok=True)
        
        self.results_file = self.output_dir / 'results.json'
        
        self.epoch_data = []
        self.mispredictions = []

    def save_predictions(self, predictions, epoch):
        """Save predictions in COCO format for TIDE"""
        pred_file = self.preds_dir / f'epoch{epoch}_predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        return pred_file

    def run_tide(self, preds_file, gt_file, epoch):
        """Run TIDE evaluation"""
        print(f"Running TIDE for epoch {epoch}...")
        
        try:
            # Load ground truth as TIDE dataset object
            gt = tide_datasets.COCO(str(gt_file))
            
            # Load predictions as TIDE result object
            pred = tide_datasets.COCOResult(str(preds_file))
            
            # Create TIDE evaluator
            tide_eval = TIDE()  # your TIDE object
            tide_eval.evaluate_range(gt, pred, mode=TIDE.BOX)

            errors = tide_eval.get_all_errors()
            print("All errors:", errors)
            
            return {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'errors': errors
            }
            
        except Exception as e:
            print(f"TIDE evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def analyze_mispredictions(self, model, val_loader, device, epoch, max_samples=50):
        """Generate Grad-CAM for backbone and feature/attention maps for FPN & ROI head."""
        print(f"Analyzing mispredictions for epoch {epoch}...")

        target_layers = [
            model.backbone,                 # Grad-CAM
            model.backbone.fpn,             # FPN feature maps
            model.roi_heads                  # ROI head features
        ]
        layer_names = ["backbone", "fpn", "roi_head"]
        mispredicted = []
        count = 0

        # Create epoch folder
        epoch_dir = self.gradcam_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)

        # Create subfolders for layers
        layer_dirs = {name: epoch_dir / name for name in layer_names}
        for ld in layer_dirs.values():
            ld.mkdir(exist_ok=True)

        for images, targets in tqdm(val_loader, desc='Processing images'):
            if count >= max_samples:
                break
            if len(images) == 0:
                continue

            img_tensor = images[0].to(device)
            target = {k: v.to(device) for k, v in targets[0].items()}

            with torch.no_grad():
                output = model([img_tensor])[0]  # single image

            # Only process mispredicted images
            image_errors = get_errors_by_image([output], [target])
            if len(image_errors) == 0:
                continue

            gt_boxes = target['boxes']
            gt_labels = target['labels']
            image_id = int(target['image_id'].item())

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            cam_paths = {}

            # ----------------- Backbone Grad-CAM -----------------
            try:
                cam_wrapper = GradCAMWrapperFRCNN(model, target_layer=model.backbone, name="backbone", device=device)
                cams = cam_wrapper.generate_cams(img_tensor.unsqueeze(0), gt_boxes, gt_labels)
                cam = cams["backbone"]
                if cam is not None:
                    if cam.shape != img_np.shape[:2]:
                        from scipy.ndimage import zoom
                        zoom_factors = (img_np.shape[0] / cam.shape[0], img_np.shape[1] / cam.shape[1])
                        cam = zoom(cam, zoom_factors, order=1)
                    visualization = show_cam_on_image(img_np, cam, use_rgb=True)
                    save_path = layer_dirs["backbone"] / f'img_{image_id}.png'
                    Image.fromarray(visualization).save(save_path)
                    cam_paths["backbone"] = str(save_path.relative_to(self.output_dir))
            except Exception as e:
                print(f"Failed Grad-CAM for image {image_id}: {e}")
                cam_paths["backbone"] = None

            # ----------------- FPN Feature Maps -----------------
            try:
                # Forward through backbone first
                fpn_maps = model.backbone(img_tensor.unsqueeze(0))  # returns dict of body features
    
                for level_name, fmap in fpn_maps.items():
                    # Average across channels
                    fmap_np = fmap[0].mean(0).detach().cpu().numpy()
                    # Normalize to [0,1]
                    fmap_np = (fmap_np - fmap_np.min()) / (fmap_np.max() - fmap_np.min() + 1e-8)
                    # Convert to PIL image
                    fmap_img = Image.fromarray((fmap_np * 255).astype(np.uint8)).resize(img_np.shape[:2][::-1])
                    # Save
                    save_path = layer_dirs["fpn"] / f"{level_name}_img_{image_id}.png"
                    fmap_img.save(save_path)

            except Exception as e:
                print(f"Failed to extract FPN maps for image {image_id}: {e}")

            # ----------------- ROI Head Feature Maps -----------------
            try:
                # 1. Transform input
                images, targets = model.transform([img_tensor], [target])

                # 2. Backbone → FPN maps
                fpn_maps = model.backbone(images.tensors)

                # 3. RPN proposals
                proposals, _ = model.rpn(images, fpn_maps)

                # 4. ROI pooled features
                pooled_features = model.roi_heads.box_roi_pool(
                    fpn_maps,              # dict[str, Tensor]
                    proposals,             # list[Tensor]
                    images.image_sizes     # list[(H, W)]
                )

                # 5. Save first pooled feature map
                if pooled_features.shape[0] > 0:
                    fmap = pooled_features[0].mean(0).detach().cpu().numpy()
                    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

                    fmap_img = Image.fromarray((fmap * 255).astype(np.uint8))
                    fmap_img.save(layer_dirs["roi_head"] / f"img_{image_id}.png")
                else:
                    print(f"No proposals found for image {image_id}")

            except Exception as e:
                print(f"Failed to extract ROI head map for image {image_id}: {e}")

                
            mispred_entry = {
                'epoch': epoch,
                'image_id': image_id,
                'errors': [e['type'] for e in image_errors[image_id]],
                'gradcam_paths': cam_paths,
                'confidences': output['scores'].cpu().tolist(),
                'predicted_classes': output['labels'].cpu().tolist(),
                'boxes': output['boxes'].cpu().tolist()
            }
            mispredicted.append(mispred_entry)
            count += 1

        print(f"Found {len(mispredicted)} error samples")
        return mispredicted

    def save_epoch_data(self, metrics, tide_result, mispredictions, epoch):
        """Save results per epoch instead of overwriting a single file."""
        
        epoch_file = self.output_dir / f'results_epoch_{epoch}.json'
        
        output = {
            'epoch': epoch,
            'metrics': metrics,
            'tide': tide_result if tide_result else None,
            'mispredictions': mispredictions,
            'metadata': {
                'saved_at': datetime.now().isoformat()
            }
        }
        
        with open(epoch_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved results for epoch {epoch} to {epoch_file}")

# -------------------------
# Get COCO predictions
# -------------------------
def get_coco_predictions(model, dataloader, device):
    """Convert model outputs to COCO format"""
    model.eval()
    results = []
    
    for images, targets in tqdm(dataloader, desc="Generating predictions"):
        if len(images) == 0:
            continue
            
        images = [img.to(device) for img in images]
        
        with torch.no_grad():
            outputs = model(images)
        
        for output, target in zip(outputs, targets):
            img_id = int(target['image_id'].item())
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
    
    return results

# -------------------------
# Compute COCO metrics
# -------------------------
def evaluate_coco(predictions, val_annotations):
    """Compute standard COCO metrics"""
    print("Computing COCO metrics...")
    
    coco_gt = COCO(val_annotations)
    
    if len(predictions) == 0:
        print("Warning: No predictions to evaluate")
        return {
            'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0,
            'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.0
        }
    
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        'mAP': float(coco_eval.stats[0]),       # AP @ IoU=0.50:0.95
        'mAP50': float(coco_eval.stats[1]),     # AP @ IoU=0.50
        'mAP75': float(coco_eval.stats[2]),     # AP @ IoU=0.75
        'mAP_small': float(coco_eval.stats[3]),
        'mAP_medium': float(coco_eval.stats[4]),
        'mAP_large': float(coco_eval.stats[5]),
        'AR_1': float(coco_eval.stats[6]),      # AR @ maxDets=1
        'AR_10': float(coco_eval.stats[7]),     # AR @ maxDets=10
        'AR_100': float(coco_eval.stats[8]),    # AR @ maxDets=100
    }
    
    print(f"\n{'='*60}")
    print("COCO Metrics (Raw Performance):")
    print(f"  mAP (IoU 0.50:0.95): {metrics['mAP']:.4f}")
    print(f"  mAP50 (IoU 0.50):    {metrics['mAP50']:.4f}")
    print(f"  mAP75 (IoU 0.75):    {metrics['mAP75']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics

    
# -------------------------
# Main training loop
# -------------------------
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Validate paths
    assert Path(args.train_annotations).exists(), f"Train annotations not found: {args.train_annotations}"
    assert Path(args.val_annotations).exists(), f"Val annotations not found: {args.val_annotations}"
    assert Path(args.train_images).exists(), f"Train images not found: {args.train_images}"
    assert Path(args.val_images).exists(), f"Val images not found: {args.val_images}"

    # Datasets & loaders
    print("Loading datasets...")
    train_dataset = CocoDetection(args.train_images, args.train_annotations)
    val_dataset = CocoDetection(args.val_images, args.val_annotations)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )

    # Model
    print("Initializing model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    
    # Optimizer & scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=0.001
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

    start_epoch = 1
    # -------------------------
    # Resume from checkpoint
    # -------------------------
    checkpoint_dir = Path(args.output)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pth"), key=lambda x: int(x.stem.split("_")[-1]))
    if checkpoints:
        last_ckpt = checkpoints[-1]
        print(f"Resuming from checkpoint: {last_ckpt}")
        checkpoint_state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint_state)
        start_epoch = int(last_ckpt.stem.split("_")[-1]) + 1
        print(f"Resuming from epoch {start_epoch}")

    analyzer = AnalyzerFRCNN(output_dir=args.output)

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for images, targets in tqdm(train_loader, desc="Training"):
            if len(images) == 0:
                continue
                
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            num_batches += 1
        
        lr_scheduler.step()
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Average training loss: {avg_loss:.4f}")

        # -------------------------
        # Save & analyze every 5 epochs
        # -------------------------
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            # Save model checkpoint
            model_file = checkpoint_dir / f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_file)
            print(f"Saved model checkpoint: {model_file}")

            # Validation
            print("\nRunning validation...")
            predictions = get_coco_predictions(model, val_loader, device)
            
            # Analyzer
            analyzer = AnalyzerFRCNN(output_dir=args.output)

            # Save predictions for TIDE
            pred_file = analyzer.save_predictions(predictions, epoch)
            
            # Compute metrics
            metrics = evaluate_coco(predictions, args.val_annotations)
            print(f"mAP: {metrics['mAP']:.4f}, mAP50: {metrics['mAP50']:.4f}")
            
            # Run TIDE
            tide_result = analyzer.run_tide(
                preds_file=pred_file,
                gt_file=args.val_annotations,
                epoch=epoch
            )
            
            # Analyze mispredictions
            mispredictions = analyzer.analyze_mispredictions(
                model, val_loader, device, epoch, max_samples=args.max_samples
            )
            
            # Save all results
            analyzer.save_epoch_data(metrics, tide_result, mispredictions, epoch)

            # Reset training 
            model.train()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*60)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Faster R-CNN TIDE Analysis Pipeline')
    
    parser.add_argument("--epochs", type=int, default=5, help='Number of training epochs')
    parser.add_argument("--batch-size", type=int, default=2, help='Training batch size')
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    
    parser.add_argument("--train-images", type=str, required=True, help='Path to training images')
    parser.add_argument("--train-annotations", type=str, required=True, help='Path to training annotations')
    parser.add_argument("--val-images", type=str, required=True, help='Path to validation images')
    parser.add_argument("--val-annotations", type=str, required=True, help='Path to validation annotations')
    
    parser.add_argument("--output", type=str, default="./tide_analysis", help='Output directory')
    parser.add_argument("--max-samples", type=int, default=50, help='Max error samples to analyze per epoch')
    
    args = parser.parse_args()
    main(args)