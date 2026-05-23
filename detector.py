"""YOLO inference wrapper. Detection type defined here so other modules can import."""

from dataclasses import dataclass


Bbox = tuple[int, int, int, int]


@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: Bbox
    confidence: float
