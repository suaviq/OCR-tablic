import cv2
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer
from util import parse_prediction

yolo_model = YOLO("../yolo_plate.pt")
reader = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')

def detect_and_read_plate(frame):
    results = yolo_model.predict(frame, verbose=False)[0]
    if not results.boxes:
        return "Brak tablicy", None

    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0].tolist())
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return "Błędne współrzędne", None

    plate_img = frame[y1:y2, x1:x2]

    if plate_img is None or plate_img.size == 0:
        return "Pusty wycinek", None
    
    plate_img = frame[y1:y2, x1:x2]
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray_plate = gray_plate[..., None]

    pred = reader.run(gray_plate)
    if not pred or not isinstance(pred, list):
        return "Błąd OCR", (x1, y1, x2, y2)
    
    text = parse_prediction(pred)
    print(f"Rozpoznana tablica: {text}")
    return text, (x1, y1, x2, y2)

def live_video_ocr(type = 'video', video_path = None):
    if type == "video":
        if not video_path:
            print("Podaj ścieżkę do pliku wideo.")
            return
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter("videos/output_annotated_video.mp4", fourcc, fps, (width, height))
    else:
        cap = cv2.VideoCapture(0)

    print(f"Tryb: {'kamera na żywo' if type == 'live' else 'plik wideo'} – rozpoczęto analizę (naciśnij 'q' by zakończyć).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plate_text, bbox = detect_and_read_plate(frame)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{plate_text}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("OCR Tablic", frame)

        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
