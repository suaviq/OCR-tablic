from pathlib import Path
import cv2
from ultralytics import YOLO
from time import time
from tqdm import tqdm

from util import calculate_iou, parse_prediction, calc_accuracy

def train_model(yaml_path):
    model = YOLO("yolo11s.pt")
    model.train(data=str(yaml_path), epochs=50, batch=16)
    model.save("yolo_plate.pt")

def run_reader(reader, annotations):
    correct = 0
    total = 0
    ious = []
    start = time()
    model = YOLO("yolo_plate.pt")

    val_labels = list(Path("yolo_data/labels/val").glob("*.txt"))

    for label_path in tqdm(val_labels[:100]):
        img_path = Path("yolo_data/images/val") / f"{label_path.stem}.jpg"
        result = model.predict(str(img_path), verbose=False)[0]
        true_box = list(map(float, open(label_path).read().strip().split()[1:]))

        ann = [ann for ann in annotations if str(ann.img_path.stem) == str(label_path.stem)][0]

        w_img, h_img = cv2.imread(str(img_path)).shape[1::-1]
        cx, cy, w, h = true_box
        x1 = (cx - w/2) * w_img
        y1 = (cy - h/2) * h_img
        x2 = (cx + w/2) * w_img
        y2 = (cy + h/2) * h_img
        gt_box = [x1, y1, x2, y2]

        if len(result.boxes) > 0:
            pred_box = result.boxes.xyxy[0].tolist()
            iou = calculate_iou(gt_box, pred_box)
            ious.append(iou)
            x1, y1, x2, y2 = map(int, pred_box)
            plate_img = cv2.imread(str(img_path))[y1:y2, x1:x2]
            gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray_plate_img = gray_plate_img[..., None]
            text_output = reader.run(gray_plate_img)
            if not text_output or not isinstance(text_output, list):
                text = 'Błąd'
            else:
                text = parse_prediction(text_output)
            if text == ann.correct:
                correct += 1  
            else:
                print(f"Predicted: {text}, Correct: {ann.correct}")
        total += 1

    end = time()
    processing_time = end - start
    accuracy = (correct / total) * 100
    mean_iou = sum(ious) / len(ious) if ious else 0

    print("")
    print(f"OCR Accuracy: {accuracy:.2f}%")
    print(f"Processing time (100 images): {processing_time:.2f} seconds")
    print(f"Mean IoU: {mean_iou:.3f}")

    return (accuracy, processing_time)
