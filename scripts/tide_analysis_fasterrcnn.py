import json
import argparse
import os
from collections import defaultdict
from torchvision.datasets import VOCDetection
import numpy as np

# Install tidecv if needed: pip install tidecv
try:
    from tidecv import TIDE, datasets
except ImportError:
    print("ERROR: tidecv not installed. Install with: pip install tidecv")
    exit(1)


def load_voc_ground_truth(data_root='./data', year='2012', image_set='val'):
    """
    Load ground truth annotations from VOC dataset.
    Returns a dictionary mapping image_id to list of ground truth boxes.
    """
    print(f"Loading VOC {year} {image_set} ground truth...")
    
    voc_ds = VOCDetection(root=data_root, year=year, image_set=image_set, download=False)
    
    # VOC class names (same as in evaluation script)
    classes = [
        "__background__", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    gt_data = {}
    
    for idx in range(len(voc_ds)):
        _, target = voc_ds[idx]
        image_id = target['annotation']['filename']
        
        boxes = []
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        
        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            
            class_name = obj['name']
            class_id = class_to_idx[class_name]
            
            boxes.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'class': class_id,
                'class_name': class_name
            })
        
        gt_data[image_id] = boxes
    
    print(f"Loaded ground truth for {len(gt_data)} images")
    return gt_data, classes


def convert_to_tide_format(predictions_file, gt_data, classes):
    """
    Convert our prediction format to TIDE format.
    
    TIDE expects:
    - Predictions: list of dicts with 'image_id', 'category_id', 'bbox', 'score'
    - Ground truth: list of dicts with 'image_id', 'category_id', 'bbox'
    - bbox format: [x, y, width, height] (COCO format)
    """
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    tide_predictions = []
    tide_gt = []
    
    # Convert predictions
    for pred in predictions:
        image_id = pred['image_id']
        
        for detection in pred['predictions']:
            bbox = detection['bbox']  # [xmin, ymin, xmax, ymax]
            # Convert to COCO format [x, y, width, height]
            x, y, xmax, ymax = bbox
            width = xmax - x
            height = ymax - y
            
            tide_predictions.append({
                'image_id': image_id,
                'category_id': detection['label'],
                'bbox': [x, y, width, height],
                'score': detection['score']
            })
    
    # Convert ground truth
    for image_id, boxes in gt_data.items():
        for box in boxes:
            bbox = box['bbox']  # [xmin, ymin, xmax, ymax]
            # Convert to COCO format [x, y, width, height]
            x, y, xmax, ymax = bbox
            width = xmax - x
            height = ymax - y
            
            tide_gt.append({
                'image_id': image_id,
                'category_id': box['class'],
                'bbox': [x, y, width, height]
            })
    
    print(f"Converted {len(tide_predictions)} predictions and {len(tide_gt)} ground truth boxes")
    return tide_predictions, tide_gt


def export_tide_json(tide, output_file, classes, num_images, dataset='Pascal VOC 2012', model='Faster R-CNN ResNet50-FPN'):
    """Export TIDE results to a structured JSON file for the dashboard.
    The JSON matches the shape expected by the React app.
    """
    # Overall metrics (AP at IoU=0.5 and MAP)
    overall_metrics = {}
    if hasattr(tide, 'ap'):
        # tide.ap may be a single value or a dict; handle both
        try:
            overall_metrics['ap_50'] = float(tide.ap)
            overall_metrics['map'] = float(tide.ap)
        except Exception:
            overall_metrics['ap_50'] = 0.0
            overall_metrics['map'] = 0.0
    else:
        all_errors = tide.get_all_errors() if hasattr(tide, 'get_all_errors') else {}
        overall_metrics['ap_50'] = float(all_errors.get('AP_50', 0.0))
        overall_metrics['map'] = float(all_errors.get('AP', 0.0))

    # Main error breakdown (dAP for each error type)
    main_errors = tide.get_main_errors() if hasattr(tide, 'get_main_errors') else {}
    main_errors = {k: float(v) for k, v in main_errors.items()}

    # Special error breakdown (false positive / false negative dAP)
    special_errors_raw = tide.get_special_errors() if hasattr(tide, 'get_special_errors') else {}
    special_errors = {
        'false_positive': float(special_errors_raw.get('FalsePos', 0.0)),
        'false_negative': float(special_errors_raw.get('FalseNeg', 0.0))
    }

    dashboard_data = {
        'metadata': {
            'dataset': dataset,
            'model': model,
            'num_images': num_images,
            'num_classes': len(classes),
            'classes': classes
        },
        'overall_metrics': overall_metrics,
        'main_errors': main_errors,
        'special_errors': special_errors
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"Dashboard data exported successfully to {output_file}")


def run_tide_analysis(predictions_file, output_dir='./results/tide_results', data_root='./data', 
                      year='2012', image_set='val'):
    """
    Run TIDE analysis on Faster R-CNN predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    gt_data, classes = load_voc_ground_truth(data_root, year, image_set)
    
    # Convert to TIDE format
    tide_preds, tide_gt = convert_to_tide_format(predictions_file, gt_data, classes)
    
    # Create TIDE evaluator
    print("\nRunning TIDE analysis...")
    tide = TIDE()
    # Debug: inspect first run object
    if tide.runs:
        first_run = tide.runs[0]
        print('First run type:', type(first_run))
        print('First run attributes:', [a for a in dir(first_run) if not a.startswith('_')])
        # Print some known attributes if they exist
        for attr in ['ap', 'main_errors', 'special_errors', 'run_info']:
            if hasattr(first_run, attr):
                print(f'first_run.{attr}:', getattr(first_run, attr))
    else:
        print('No runs found in TIDE object')

    print('TIDE runs:', tide.runs)
    # Debug: print available attributes for export
    print('TIDE attributes:', [a for a in dir(tide) if not a.startswith('_')])
    
    # Create Data objects for TIDE
    from tidecv.data import Data
    
    # Create ground truth Data object
    gt_data_obj = Data('VOC_GT')
    for gt in tide_gt:
        # add_ground_truth expects: image_id, category_id, bbox
        gt_data_obj.add_ground_truth(
            gt['image_id'],
            gt['category_id'],
            gt['bbox']
        )
    
    # Create predictions Data object  
    pred_data_obj = Data('VOC_Pred')
    for pred in tide_preds:
        # add_detection expects: image_id, category_id, score, bbox
        pred_data_obj.add_detection(
            pred['image_id'],
            pred['category_id'],
            pred['score'],
            pred['bbox']
        )
    
    # Run evaluation
    print("Evaluating with TIDE...")
    tide.evaluate(gt_data_obj, pred_data_obj, mode=TIDE.BOX)
    
    # Generate summary
    print("\n" + "="*60)
    print("TIDE Error Analysis Summary")
    print("="*60)
    tide.summarize()
    
    # Save text summary
    results_file = os.path.join(output_dir, 'tide_summary.txt')
    print(f"\nSaving text summary to {results_file}...")
    
    import sys
    original_stdout = sys.stdout
    with open(results_file, 'w') as f:
        sys.stdout = f
        tide.summarize()
        sys.stdout = original_stdout
    
    # Export structured JSON for dashboard
    json_file = os.path.join(output_dir, 'tide_data.json')
    print(f"Exporting structured data to {json_file}...")
    export_tide_json(tide, json_file, classes)
    
    # Generate plots
    print(f"Generating plots to {output_dir}...")
    tide.plot(out_dir=output_dir)
    
    print("\n" + "="*60)
    print("TIDE analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TIDE analysis on Faster R-CNN VOC predictions')
    parser.add_argument('--predictions', type=str, default='./results/predictions.json',
                        help='Path to predictions JSON file from evaluate_fasterrcnn_voc.py')
    parser.add_argument('--output_dir', type=str, default='./results/tide_results',
                        help='Directory to save TIDE analysis results')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory of VOC dataset')
    parser.add_argument('--year', type=str, default='2012',
                        help='VOC dataset year')
    parser.add_argument('--image_set', type=str, default='val',
                        help='Image set to use (train/val)')
    
    args = parser.parse_args()
    
    run_tide_analysis(
        predictions_file=args.predictions,
        output_dir=args.output_dir,
        data_root=args.data_root,
        year=args.year,
        image_set=args.image_set
    )
