"""Constants for SH17 PPE Compliance Checker.

Class IDs follow sh17.yaml (the order used by SH17 pre-trained weights),
NOT the order printed in the SH17 README.
"""

CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}

PERSON_CLASS_ID: int = 0
PERSON_GROUP_CLASS_IDS: set[int] = {0, 3, 12}

RULE_PRESETS: dict[str, list[str]] = {
    "HEAD": ["helmet"],
    "BODY": ["safety-vest", "safety-suit", "medical-suit"],
    "HAND": ["gloves"],
    "FOOT": ["shoes"],
}

PPE_TO_GROUP: dict[str, str] = {
    name: group for group, names in RULE_PRESETS.items() for name in names
}

DEFAULT_CONF_PERSON: float = 0.5
DEFAULT_CONF_PPE: float = 0.3
CONTAINMENT_THRESHOLD: float = 0.7

WEIGHTS_PATH: str = "weights/yolo9s.pt"
MAX_DISPLAY_WIDTH: int = 1024
MAX_INFERENCE_WIDTH: int = 4096

GROUP_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "person": (255, 128, 0),
    "head_face": (180, 180, 180),
    "ppe_found": (0, 200, 0),
    "ppe_orphan": (0, 200, 220),
    "other": (160, 160, 160),
}
