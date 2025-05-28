from pathlib import Path
from dataclasses import dataclass
import os
import random
import shutil
import xml.etree.ElementTree as ET

@dataclass
class Annotation:
    """Klasa przedstawiająca annotację zdjęcia z pliku `annotations.xml`"""
    img_path: Path
    img_size: tuple[int, int]
    bounding_box: tuple[int, int, int, int]
    correct: str = ""


def parse_annotations(xml_path: str, images_dir: str) -> list[Annotation]:
    annotations = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image_tag in root.findall("image"):
        filename = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))
        img_path = Path(images_dir) / filename

        box_tag = image_tag.find("box")
        if box_tag is not None:
            xtl = int(float(box_tag.get("xtl")))
            ytl = int(float(box_tag.get("ytl")))
            xbr = int(float(box_tag.get("xbr")))
            ybr = int(float(box_tag.get("ybr")))
            attr_tag = box_tag.find("attribute")
            plate_num = str(attr_tag.text)
            annotation = Annotation(img_path, (width, height), (xtl, ytl, xbr, ybr))
            annotation.correct = plate_num
            annotations.append(annotation)
    return annotations


def normalize_bounding_box(bbox, img_size):
    xmin, ymin, xmax, ymax = bbox
    width, height = img_size
    cx = ((xmin + xmax) / 2) / width
    cy = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return cx, cy, w, h


def train_test_split(data, val_size=100):
    random.shuffle(data)
    if isinstance(val_size, float):  
        val_size = int(len(data) * val_size)
    train_size = len(data) - val_size
    return data[:train_size], data[train_size:]


def prepare_yolo_dataset(annotations, output_dir="yolo_data", val_size=100):
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(path / folder, exist_ok=True)

    train, val = train_test_split(annotations, val_size=val_size)
    
    for split_name, split_data in zip(["train", "val"], [train, val]):
        for ann in split_data:
            img_out = path / f"images/{split_name}/{ann.img_path.name}"
            label_out = path / f"labels/{split_name}/{ann.img_path.stem}.txt"
            shutil.copy(ann.img_path, img_out)
            cx, cy, w, h = normalize_bounding_box(ann.bounding_box, ann.img_size)
            with open(label_out, "w") as f:
                f.write(f"0 {cx} {cy} {w} {h}\n")

    with open(path / "data.yaml", "w") as f:
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['plate']\n")
    
    return path / "data.yaml"
