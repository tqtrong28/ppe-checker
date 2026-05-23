"""YOLO inference wrapper."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import CLASS_NAMES, PERSON_GROUP_CLASS_IDS


Bbox = tuple[int, int, int, int]


@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: Bbox
    confidence: float


class Detector:
    def __init__(self, weights_path: str):
        if not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"Weight file not found at {weights_path}. "
                "Download from https://github.com/ahmadmughees/SH17dataset/releases/tag/v1"
            )
        self.model = YOLO(weights_path)

    def predict(
        self,
        image: np.ndarray,
        conf_person: float,
        conf_ppe: float,
    ) -> list[Detection]:
        """Run inference and return detections filtered by two confidence thresholds.

        - Classes in PERSON_GROUP_CLASS_IDS (person, face, head) use `conf_person`.
        - All other classes use `conf_ppe`.
        """
        min_conf = min(conf_person, conf_ppe)
        results = self.model(image, conf=min_conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                threshold = conf_person if cls_id in PERSON_GROUP_CLASS_IDS else conf_ppe
                if conf < threshold:
                    continue
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=CLASS_NAMES.get(cls_id, str(cls_id)),
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                    )
                )
        return detections
