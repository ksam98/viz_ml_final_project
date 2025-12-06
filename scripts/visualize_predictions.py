import json
import cv2
import os
import argparse
import matplotlib.pyplot as plt

def visualize_predictions(image_id, predictions_file, image_dir='./dummy_pred', threshold=0.3):
    """
    Visualize all bounding boxes for a specific image ID from a predictions JSON file.
    """
    # 1. Load Predictions
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        all_predictions = json.load(f)
    
    # Filter predictions for the specific image_id
    image_preds = [
        p for p in all_predictions 
        if str(p['image_id']) == str(image_id)
    ]
    
    if not image_preds:
        print(f"No predictions found for image ID {image_id}")
        return

    print(f"Found {len(image_preds)} predictions for image {image_id}")

    # 2. Load the Image
    # Construct filename: 12 digits, zero padded
    # e.g. 52591 -> 000000052591.jpg
    image_filename = f"{int(image_id):012d}.jpg"
    image_path = os.path.join(image_dir, image_filename)
    
    print(f"Loading image from {image_path}...")
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    target_image = cv2.imread(image_path)
    if target_image is None:
        print(f"Failed to load image: {image_path}")
        return

    # VOC class names (hardcoded for now as we aren't loading the dataset)
    classes = [
        "__background__", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    
    # 3. Draw Bounding Boxes
    # Create a copy to draw on
    vis_image = target_image.copy()
    
    count_visible = 0
    for pred in image_preds:
        bbox = pred['bbox']
        score = pred['score']
        category_id = pred['category_id']
        
        # Threshold to avoid cluttering with very low confidence boxes
        if score < threshold:
            continue
        
        count_visible += 1
            
        # bbox format is [x, y, width, height]
        x, y, w, h = map(int, bbox)
        
        # Get class name safely
        class_name = classes[category_id] if category_id < len(classes) else f"Class {category_id}"
        
        # Draw rectangle
        # Color: Green for high confidence, Red for low
        color = (0, 255, 0) if score > 0.7 else (0, 0, 255)
        
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Draw Label
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(vis_image, label_text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"Visualized {count_visible} predictions (threshold: {threshold})")

    # 4. Show Image
    # Convert back to RGB for Matplotlib
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_image_rgb)
    plt.axis('off')
    plt.title(f"Image {image_id} - Predictions from {os.path.basename(predictions_file)}")
    plt.show()
    
    # Also save it for easy viewing without UI
    # output_filename = f"vis_{image_id}.jpg"
    # cv2.imwrite(output_filename, vis_image)
    # print(f"Visualization saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize predictions on an image')
    parser.add_argument('--image_id', type=str, required=True, help='Image ID (e.g., 52591)')
    parser.add_argument('--json_file', type=str, required=True, help='Path to predictions JSON file')
    parser.add_argument('--image_dir', type=str, default='./dummy_pred', help='Directory containing images')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold for visualization')
    
    args = parser.parse_args()
    
    visualize_predictions(args.image_id, args.json_file, args.image_dir, args.threshold)
