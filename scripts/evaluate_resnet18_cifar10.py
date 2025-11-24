import torch
from datasets import load_dataset
from transformers import (
    ResNetForImageClassification,
    ResNetConfig,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    CenterCrop,
)
import json
import os
import glob
import argparse
from tqdm import tqdm
import numpy as np

def evaluate(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading CIFAR-10 test dataset...")
    dataset = load_dataset("cifar10")
    test_ds = dataset["test"]
    
    # Get class names
    class_names = dataset["train"].features["label"].names
    
    # 2. Preprocessing
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    size = 64  # Same as training
    
    _test_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def test_transforms(examples):
        examples["pixel_values"] = [_test_transforms(image.convert("RGB")) for image in examples["img"]]
        return examples

    # 3. Load Model
    checkpoint_dirs = glob.glob(os.path.join(args.checkpoint_dir, "resnet18_cifar10_epoch_*"))
    
    if checkpoint_dirs:
        # Sort by epoch number and get the latest
        checkpoint_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        model = ResNetForImageClassification.from_pretrained(latest_checkpoint)
        epoch_num = int(latest_checkpoint.split('_')[-1])
        print(f"Loaded model from epoch {epoch_num}")
    else:
        print("No checkpoint found. Loading base pre-trained model...")
        model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-18",
            num_labels=len(class_names),
            ignore_mismatched_sizes=True,
        )
        print("Loaded base ResNet-18 model pre-trained on ImageNet")
    
    model.to(device)
    model.eval()

    # 4. Run Inference
    print(f"Running inference on test set (max {args.max_images if args.max_images else 'all'} images)...")
    predictions = []
    
    # Determine how many images to process
    num_images = min(args.max_images, len(test_ds)) if args.max_images else len(test_ds)
    
    with torch.no_grad():
        for idx in tqdm(range(num_images), desc="Evaluating"):
            # Get single example
            example = test_ds[idx]
            image = example["img"]
            true_label = example["label"]
            
            # Transform image
            pixel_values = _test_transforms(image.convert("RGB")).unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(pixel_values)
            logits = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probs[0, predicted_class_idx].item()
            
            # Get all class probabilities
            all_probs = probs[0].cpu().numpy().tolist()
            
            prediction = {
                'image_index': idx,
                'true_label': int(true_label),
                'true_class_name': class_names[true_label],
                'predicted_label': int(predicted_class_idx),
                'predicted_class_name': class_names[predicted_class_idx],
                'confidence': float(confidence),
                'correct': predicted_class_idx == true_label,
                'class_probabilities': {
                    class_names[i]: float(prob) for i, prob in enumerate(all_probs)
                }
            }
            
            predictions.append(prediction)

    # 5. Save predictions to JSON
    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\nPredictions saved to: {output_file}")
    
    # 6. Compute and display statistics
    total_images = len(predictions)
    correct_predictions = sum(1 for pred in predictions if pred['correct'])
    accuracy = correct_predictions / total_images * 100
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Total images evaluated: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    
    for pred in predictions:
        true_class = pred['true_class_name']
        class_total[true_class] += 1
        if pred['correct']:
            class_correct[true_class] += 1
    
    print(f"\nPer-class accuracy:")
    for class_name in class_names:
        if class_total[class_name] > 0:
            class_acc = class_correct[class_name] / class_total[class_name] * 100
            print(f"  {class_name:15s}: {class_acc:5.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
    
    # Average confidence
    avg_confidence = np.mean([pred['confidence'] for pred in predictions])
    avg_confidence_correct = np.mean([pred['confidence'] for pred in predictions if pred['correct']])
    avg_confidence_incorrect = np.mean([pred['confidence'] for pred in predictions if not pred['correct']]) if any(not pred['correct'] for pred in predictions) else 0
    
    print(f"\nConfidence statistics:")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Average confidence (correct): {avg_confidence_correct:.4f}")
    if avg_confidence_incorrect > 0:
        print(f"  Average confidence (incorrect): {avg_confidence_incorrect:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 on CIFAR-10')
    parser.add_argument('--checkpoint_dir', type=str, default='./cifar10_resnet18_checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_file', type=str, default='./cifar10_predictions.json',
                        help='Output JSON file for predictions')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to evaluate (default: all)')
    args = parser.parse_args()
    evaluate(args)
