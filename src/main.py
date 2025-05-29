from fast_plate_ocr import ONNXPlateRecognizer
from annotation import parse_annotations, prepare_yolo_dataset
from reader import train_model, run_reader
from live_video import live_video_ocr

RUN_TRAIN = False
LIVE_VIDEO_MODE = True
VALIDATION = False

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2

if __name__ == '__main__':
    if VALIDATION:
        annotations = parse_annotations("./dataset/annotations.xml", "./dataset/photos")
        yaml_path = prepare_yolo_dataset(annotations)

        if RUN_TRAIN:
            train_model(yaml_path)

        fastplate_reader = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')
        (fastplate_accuracy, fastplate_processing_time) = run_reader(fastplate_reader, annotations)

        print('Final grade using EasyOCR: ' + str(calculate_final_grade(fastplate_accuracy, fastplate_processing_time)))
    
    if LIVE_VIDEO_MODE:
        live_video_ocr(type="video", video_path="../videos/example_video.mp4")
        # live_video_ocr(type="live")

