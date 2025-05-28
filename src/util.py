from difflib import SequenceMatcher


def calc_accuracy(correct: str, read: str) -> float:
    return SequenceMatcher(None, correct, read).ratio()

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def parse_prediction(prediction: list[str]) -> str:
    if len(prediction) == 0:
        return 'Błąd'
    if prediction[0] == "PL":
        prediction.pop(0)
    if len(prediction) == 0:
        return 'Błąd'

    parsed = prediction[0]
    parsed = parsed.replace(' ', '').replace('_', '').replace('|', 'I')

    return parsed
