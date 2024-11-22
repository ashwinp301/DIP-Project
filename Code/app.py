import os
import json
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# For Faster R-CNN
class_names_all = {
    0: "background",
    1: "Grenade",
    2: "Handgun",
    3: "Knife",
    4: "Rifle",
    5: "Sword",
}

# For YOLO (no background class)
class_names_no_background_yolo = {
    1: "Grenade",
    2: "Handgun",
    3: "Knife",
    4: "Rifle",
    5: "Sword",
}

# Set up device: CUDA for GPU support, or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Load models
yolo_model = YOLO('models/best_new.pt')  # Load YOLOv11 model

# Set the number of classes for Faster R-CNN to match the saved model
faster_rcnn_classes = 6

# Load Faster R-CNN and update the classifier head for the correct number of classes
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, faster_rcnn_classes)
faster_rcnn_model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, faster_rcnn_classes * 4)
faster_rcnn_model.load_state_dict(torch.load('models/faster_rcnn_model_DIP_newDataset.pth', map_location=device))
faster_rcnn_model.to(device)

# Define transformations for input images
transform = transforms.Compose([transforms.ToTensor()])

# Route to handle image upload and prediction
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model')  # Get the selected model from form data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    upload_folder = app.config['UPLOAD_FOLDER']
    filepath = os.path.join(upload_folder, filename)
    
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(os.path.join(upload_folder, 'annotated'), exist_ok=True)

    file.save(filepath)  # Save the uploaded file

    # Open image to get its dimensions
    image = Image.open(filepath)
    image_width, image_height = image.size

    # Perform prediction based on the selected model
    if model_choice == 'yolo' or model_choice == 'faster_rcnn' or model_choice == 'ensemble':
        yolo_results = yolov11_predict(yolo_model, filepath)
        faster_rcnn_image = image
        faster_rcnn_image_tensor = transform(faster_rcnn_image).unsqueeze(0).to(device)
        faster_rcnn_results = faster_rcnn_predict(faster_rcnn_model, faster_rcnn_image_tensor)

        # If ensemble model is selected, apply WBF
        if model_choice == 'ensemble':
            combined_results = apply_wbf(yolo_results, faster_rcnn_results, image_width, image_height)
        else:
            # For individual models, use their respective results
            combined_results = yolo_results if model_choice == 'yolo' else faster_rcnn_results

        # Annotate image with combined results
        annotated_image_path = draw_bounding_boxes(filepath, combined_results, class_names_all)
        annotated_image_url = url_for('static', filename=f'uploads/annotated/{filename}')
        
        # Return the predictions with detailed bounding box info
        response = {
            'result': combined_results,
            'image_url': annotated_image_url,
            'predictions': combined_results  # Include detailed predictions here
        }
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    return jsonify(response)


# Function to apply WBF (Weighted Box Fusion) on YOLO and Faster R-CNN results
def apply_wbf(yolo_results, faster_rcnn_results, image_width, image_height, iou_threshold=0.4):
    # Extract boxes, scores, and labels for WBF
    boxes = []
    scores = []
    labels = []

    # Process YOLO results
    for result in yolo_results:
        box = result['box']
        if is_valid_box(box):
            # Normalize boxes to [0, 1] range
            normalized_box = normalize_box(box, image_width, image_height)
            boxes.append(normalized_box)
            scores.append(result['score'])
            labels.append(result['label'])

    # Process Faster R-CNN results
    for result in faster_rcnn_results:
        box = result['box']
        if is_valid_box(box):
            # Normalize boxes to [0, 1] range
            normalized_box = normalize_box(box, image_width, image_height)
            boxes.append(normalized_box)
            scores.append(result['score'])
            labels.append(result['label'])

    # Convert lists to appropriate format for WBF
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    # Apply WBF to combine results
    boxes, scores, labels = weighted_boxes_fusion(
        [boxes], [scores], [labels], weights=None, iou_thr=iou_threshold, skip_box_thr=0.0)

    # Rescale the boxes back to the original image size
    boxes = rescale_boxes(boxes, image_width, image_height)

    # Format results for returning to frontend
    combined_results = []
    for box, score, label in zip(boxes, scores, labels):
        combined_results.append({
            'box': box.tolist(),  # Convert to list for JSON serialization
            'score': score,
            'label': label,
        })

    return combined_results

# Helper function to check if a bounding box is valid (non-zero area)
def is_valid_box(box):
    x_min, y_min, x_max, y_max = box
    return (x_max > x_min) and (y_max > y_min)

# Normalize a bounding box to [0, 1] range
def normalize_box(box, image_width, image_height):
    x_min, y_min, x_max, y_max = box
    return [
        x_min / image_width,  # Normalize x_min
        y_min / image_height, # Normalize y_min
        x_max / image_width,  # Normalize x_max
        y_max / image_height, # Normalize y_max
    ]

# Rescale the normalized bounding boxes back to the original size
def rescale_boxes(boxes, image_width, image_height):
    rescaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rescaled_box = [
            x_min * image_width,
            y_min * image_height,
            x_max * image_width,
            y_max * image_height
        ]
        rescaled_boxes.append(rescaled_box)
    return np.array(rescaled_boxes)

# YOLOv11 Prediction Function
def yolov11_predict(model, image_path, conf_threshold=0.25, iou_threshold=0.4):
    results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold)
    predictions = results[0]
    boxes = predictions.boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = predictions.boxes.conf.cpu().numpy()  # Confidence scores
    labels = predictions.boxes.cls.cpu().numpy()  # Class labels

    formatted_results = []
    for box, score, label in zip(boxes, scores, labels):
        formatted_results.append({
            'box': box.tolist(),
            'label': int(label + 1),  # Adjust for YOLO's class numbering
            'score': float(score)
        })

    return formatted_results

# Set a confidence threshold
confidence_threshold = 0.3

# Faster R-CNN Prediction (for object detection)
def faster_rcnn_predict(model, image, confidence_threshold=0.3):
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    return format_predictions(predictions, confidence_threshold)

# Helper to format predictions for object detection
def format_predictions(predictions, confidence_threshold):
    results = []
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score.item() >= confidence_threshold:  # Only include boxes with confidence above threshold
            results.append({
                'box': box.tolist(),
                'label': label.item(),
                'score': score.item()
            })
    return results

def draw_bounding_boxes(image_path, predictions, class_names):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except IOError:
        font = ImageFont.load_default()

    for pred in predictions:
        label = pred["label"]
        score = pred.get("score", None)
        class_name = class_names.get(label, f"Class {label}")
        
        if "box" in pred:
            box = pred["box"]
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            text = f"{class_name}: {score:.2f}" if score is not None else class_name
            
            # Calculate text size using textbbox
            text_bbox = draw.textbbox((x_min, y_min), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Draw the background rectangle for the text
            draw.rectangle([x_min, y_min - text_height, x_min + text_width, y_min], fill="red")
            draw.text((x_min, y_min - text_height), text, fill="white", font=font)

    annotated_image_path = image_path.replace("uploads", "uploads/annotated")
    os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
    image.save(annotated_image_path)
    
    return annotated_image_path

if __name__ == '__main__':
    app.run(debug=True)
