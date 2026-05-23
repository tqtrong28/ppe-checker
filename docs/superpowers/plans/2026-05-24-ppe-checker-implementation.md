# SH17 PPE Compliance Checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-page Streamlit web app that detects PPE compliance violations on uploaded workplace images, using pre-trained YOLOv9-s weights from the SH17 dataset, with per-person spatial attribution and configurable rules.

**Architecture:** Four-module layout. `config.py` holds constants. `detector.py` wraps ultralytics YOLO inference and post-filters by two confidence thresholds. `compliance.py` is pure logic — spatial containment + per-person rule check. `app.py` is the Streamlit UI calling the above. Tests target only `compliance.py` (pure, deterministic).

**Tech Stack:** Python 3.10+, Streamlit 1.30+, ultralytics 8.0.38, opencv-python, numpy, Pillow, pandas, pytest.

**Working directory throughout this plan:** `/Users/tranquangtrong/Desktop/ppe-checker/`

**Reference spec:** [`docs/superpowers/specs/2026-05-23-ppe-checker-design.md`](../specs/2026-05-23-ppe-checker-design.md)

---

## File Structure

```
ppe-checker/                                    (existing)
├── docs/                                       (existing)
│   └── superpowers/
│       ├── specs/2026-05-23-ppe-checker-design.md  (existing)
│       └── plans/2026-05-24-ppe-checker-implementation.md (this file)
├── .gitignore                                  Task 1
├── requirements.txt                            Task 1
├── config.py                                   Task 2
├── compliance.py                               Tasks 3-4
├── tests/
│   ├── __init__.py                             Task 3
│   └── test_compliance.py                      Tasks 3-4
├── detector.py                                 Task 5
├── weights/
│   └── yolo9s.pt                               Task 6 (manual download)
├── app.py                                      Tasks 7-10
├── samples/                                    Task 11 (manual + 3-5 images)
└── README.md                                   Task 11
```

---

## Task 1: Project bootstrap

**Files:**
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/.gitignore`
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/requirements.txt`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
git init
```

Expected: `Initialized empty Git repository in /Users/tranquangtrong/Desktop/ppe-checker/.git/`

- [ ] **Step 2: Create `.gitignore`**

File: `/Users/tranquangtrong/Desktop/ppe-checker/.gitignore`

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/
venv/
.DS_Store
.idea/
.vscode/
weights/*.pt
*.egg-info/
```

- [ ] **Step 3: Create `requirements.txt`**

File: `/Users/tranquangtrong/Desktop/ppe-checker/requirements.txt`

```
streamlit>=1.30.0
ultralytics==8.0.38
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pandas>=2.0.0
pytest>=7.4.0
```

- [ ] **Step 4: Create virtual environment and install**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: All packages install without errors. The ultralytics install will also pull torch automatically.

- [ ] **Step 5: Verify installation**

```bash
python -c "import streamlit, ultralytics, cv2, numpy, PIL, pandas, pytest; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Initial commit**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
git add .gitignore requirements.txt docs/
git commit -m "chore: bootstrap project with deps and spec"
```

---

## Task 2: Constants module (`config.py`)

**Files:**
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/config.py`

This module has no logic, just constants. No tests needed.

- [ ] **Step 1: Create `config.py`**

File: `/Users/tranquangtrong/Desktop/ppe-checker/config.py`

```python
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
```

- [ ] **Step 2: Verify it imports**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
python -c "import config; print(len(config.CLASS_NAMES), 'classes,', list(config.RULE_PRESETS.keys()))"
```

Expected: `17 classes, ['HEAD', 'BODY', 'HAND', 'FOOT']`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config module with class mapping and rule presets"
```

---

## Task 3: `compliance.containment_ratio` (TDD)

**Files:**
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/tests/__init__.py`
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/tests/test_compliance.py`
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/compliance.py`

- [ ] **Step 1: Write first failing test**

File: `/Users/tranquangtrong/Desktop/ppe-checker/tests/__init__.py`

```python
```

(empty file — marks tests/ as a package)

File: `/Users/tranquangtrong/Desktop/ppe-checker/tests/test_compliance.py`

```python
from compliance import containment_ratio


def test_containment_full_inside():
    inner = (10, 10, 20, 20)
    outer = (0, 0, 100, 100)
    assert containment_ratio(inner, outer) == 1.0
```

- [ ] **Step 2: Run and verify it fails**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
pytest tests/test_compliance.py::test_containment_full_inside -v
```

Expected: `ModuleNotFoundError: No module named 'compliance'` (or ImportError).

- [ ] **Step 3: Create minimal `compliance.py` with `containment_ratio`**

File: `/Users/tranquangtrong/Desktop/ppe-checker/compliance.py`

```python
"""Pure logic for PPE compliance checking.

Spatial association uses bounding box containment: a PPE detection is
attributed to a person if the PPE bbox is mostly inside the person bbox.
"""

from dataclasses import dataclass, field

from config import PERSON_CLASS_ID, PPE_TO_GROUP


Bbox = tuple[int, int, int, int]


def containment_ratio(inner: Bbox, outer: Bbox) -> float:
    """Fraction of `inner`'s area that lies inside `outer`.

    Returns 0.0 if `inner` has zero area.
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inner_area == 0:
        return 0.0
    xx1 = max(ix1, ox1)
    yy1 = max(iy1, oy1)
    xx2 = min(ix2, ox2)
    yy2 = min(iy2, oy2)
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    return inter / inner_area
```

- [ ] **Step 4: Run test, verify passes**

```bash
pytest tests/test_compliance.py::test_containment_full_inside -v
```

Expected: 1 passed.

- [ ] **Step 5: Add remaining containment tests**

Append to `/Users/tranquangtrong/Desktop/ppe-checker/tests/test_compliance.py`:

```python
def test_containment_half_overlap():
    inner = (0, 0, 10, 10)
    outer = (5, 0, 15, 10)
    assert containment_ratio(inner, outer) == 0.5


def test_containment_no_overlap():
    inner = (0, 0, 10, 10)
    outer = (100, 100, 200, 200)
    assert containment_ratio(inner, outer) == 0.0


def test_containment_zero_area_inner():
    inner = (10, 10, 10, 20)
    outer = (0, 0, 100, 100)
    assert containment_ratio(inner, outer) == 0.0


def test_containment_outer_inside_inner():
    inner = (0, 0, 100, 100)
    outer = (40, 40, 60, 60)
    assert containment_ratio(inner, outer) == (20 * 20) / (100 * 100)
```

- [ ] **Step 6: Run all containment tests**

```bash
pytest tests/test_compliance.py -v -k containment
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add compliance.py tests/__init__.py tests/test_compliance.py
git commit -m "feat: add containment_ratio with unit tests"
```

---

## Task 4: `compliance.check_compliance` (TDD)

**Files:**
- Modify: `/Users/tranquangtrong/Desktop/ppe-checker/compliance.py`
- Modify: `/Users/tranquangtrong/Desktop/ppe-checker/tests/test_compliance.py`

- [ ] **Step 1: Define test helpers and write first failing test**

Append to top of `/Users/tranquangtrong/Desktop/ppe-checker/tests/test_compliance.py` (just after the `from compliance import containment_ratio` line):

```python
from compliance import PersonCompliance, check_compliance


def make_det(class_id: int, class_name: str, bbox, conf=0.9):
    """Helper: build a Detection-like object for tests."""
    from detector import Detection
    return Detection(class_id=class_id, class_name=class_name, bbox=bbox, confidence=conf)
```

Then append this test:

```python
def test_one_person_fully_compliant():
    person = make_det(0, "person", (0, 0, 100, 200))
    helmet = make_det(10, "helmet", (20, 10, 60, 40))
    vest = make_det(16, "safety-vest", (20, 60, 80, 120))
    gloves = make_det(9, "gloves", (5, 90, 25, 110))
    shoes = make_det(14, "shoes", (30, 180, 70, 195))
    result = check_compliance(
        [person, helmet, vest, gloves, shoes],
        required_groups=["HEAD", "BODY", "HAND", "FOOT"],
    )
    assert len(result) == 1
    assert result[0].violations == []
    assert result[0].is_compliant is True
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_compliance.py::test_one_person_fully_compliant -v
```

Expected: ImportError on `PersonCompliance` / `check_compliance` / `Detection`.

- [ ] **Step 3: Add `Detection` stub in `detector.py`**

(We need `Detection` for tests. The full `Detector` class comes in Task 5.)

File: `/Users/tranquangtrong/Desktop/ppe-checker/detector.py`

```python
"""YOLO inference wrapper. Detection type defined here so other modules can import."""

from dataclasses import dataclass


Bbox = tuple[int, int, int, int]


@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: Bbox
    confidence: float
```

- [ ] **Step 4: Implement `PersonCompliance` and `check_compliance`**

Append to `/Users/tranquangtrong/Desktop/ppe-checker/compliance.py`:

```python
@dataclass
class PersonCompliance:
    person_idx: int
    person_bbox: Bbox
    person_conf: float
    found_ppe: dict[str, list] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        return len(self.violations) == 0


def check_compliance(
    detections,
    required_groups: list[str],
    containment_threshold: float = 0.7,
) -> list[PersonCompliance]:
    """Attribute PPE to persons and evaluate required groups per person.

    A PPE detection counts for a person if containment_ratio(ppe.bbox, person.bbox)
    is at least `containment_threshold`. Only PPE classes in RULE_PRESETS are checked;
    other classes (ear, hands, tool, etc.) are ignored.
    """
    persons = [d for d in detections if d.class_id == PERSON_CLASS_ID]
    ppe_items = [d for d in detections if d.class_name in PPE_TO_GROUP]

    results: list[PersonCompliance] = []
    for idx, person in enumerate(persons):
        found_ppe = {group: [] for group in required_groups}
        for ppe in ppe_items:
            group = PPE_TO_GROUP[ppe.class_name]
            if group not in required_groups:
                continue
            if containment_ratio(ppe.bbox, person.bbox) >= containment_threshold:
                found_ppe[group].append(ppe)
        violations = [g for g in required_groups if not found_ppe[g]]
        results.append(
            PersonCompliance(
                person_idx=idx,
                person_bbox=person.bbox,
                person_conf=person.confidence,
                found_ppe=found_ppe,
                violations=violations,
            )
        )
    return results
```

- [ ] **Step 5: Run test, verify it passes**

```bash
pytest tests/test_compliance.py::test_one_person_fully_compliant -v
```

Expected: 1 passed.

- [ ] **Step 6: Add remaining compliance tests**

Append to `/Users/tranquangtrong/Desktop/ppe-checker/tests/test_compliance.py`:

```python
def test_one_person_missing_helmet():
    person = make_det(0, "person", (0, 0, 100, 200))
    vest = make_det(16, "safety-vest", (20, 60, 80, 120))
    result = check_compliance(
        [person, vest],
        required_groups=["HEAD", "BODY"],
    )
    assert result[0].violations == ["HEAD"]
    assert result[0].is_compliant is False


def test_orphan_helmet_outside_person():
    person = make_det(0, "person", (0, 0, 100, 200))
    helmet_on_shelf = make_det(10, "helmet", (300, 300, 340, 330))
    result = check_compliance(
        [person, helmet_on_shelf],
        required_groups=["HEAD"],
    )
    assert result[0].violations == ["HEAD"]


def test_two_persons_one_compliant_one_not():
    p1 = make_det(0, "person", (0, 0, 100, 200))
    p2 = make_det(0, "person", (200, 0, 300, 200))
    helmet1 = make_det(10, "helmet", (20, 10, 60, 40))
    result = check_compliance(
        [p1, p2, helmet1],
        required_groups=["HEAD"],
    )
    assert result[0].is_compliant is True
    assert result[1].violations == ["HEAD"]


def test_body_group_satisfied_by_safety_suit():
    person = make_det(0, "person", (0, 0, 100, 200))
    suit = make_det(15, "safety-suit", (10, 30, 90, 180))
    result = check_compliance(
        [person, suit],
        required_groups=["BODY"],
    )
    assert result[0].violations == []


def test_body_group_satisfied_by_medical_suit():
    person = make_det(0, "person", (0, 0, 100, 200))
    suit = make_det(13, "medical-suit", (10, 30, 90, 180))
    result = check_compliance(
        [person, suit],
        required_groups=["BODY"],
    )
    assert result[0].violations == []


def test_no_required_groups():
    person = make_det(0, "person", (0, 0, 100, 200))
    result = check_compliance(
        [person],
        required_groups=[],
    )
    assert result[0].violations == []


def test_partial_containment_below_threshold():
    person = make_det(0, "person", (0, 0, 100, 200))
    helmet_60_percent = make_det(10, "helmet", (60, 10, 160, 40))
    result = check_compliance(
        [person, helmet_60_percent],
        required_groups=["HEAD"],
        containment_threshold=0.7,
    )
    assert result[0].violations == ["HEAD"]


def test_no_persons_in_image():
    helmet = make_det(10, "helmet", (0, 0, 50, 50))
    result = check_compliance(
        [helmet],
        required_groups=["HEAD"],
    )
    assert result == []
```

- [ ] **Step 7: Run all compliance tests**

```bash
pytest tests/test_compliance.py -v
```

Expected: 14 passed (5 containment + 9 compliance).

- [ ] **Step 8: Commit**

```bash
git add compliance.py detector.py tests/test_compliance.py
git commit -m "feat: add check_compliance with per-person PPE attribution"
```

---

## Task 5: Detector class (YOLO wrapper)

**Files:**
- Modify: `/Users/tranquangtrong/Desktop/ppe-checker/detector.py`

No unit tests for this module — it wraps an external library and requires the weight file. We'll smoke-test it manually in Task 6.

- [ ] **Step 1: Add `Detector` class to `detector.py`**

Replace the entire contents of `/Users/tranquangtrong/Desktop/ppe-checker/detector.py` with:

```python
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
```

- [ ] **Step 2: Verify `Detection` import still works**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
pytest tests/test_compliance.py -v
```

Expected: All 13 tests still pass.

- [ ] **Step 3: Verify graceful error when weights missing**

```bash
python -c "from detector import Detector; Detector('weights/yolo9s.pt')"
```

Expected: `FileNotFoundError: Weight file not found at weights/yolo9s.pt. Download from ...`

- [ ] **Step 4: Commit**

```bash
git add detector.py
git commit -m "feat: add Detector class with dual-threshold filtering"
```

---

## Task 6: Download weights and smoke-test detector

**Files:**
- Manual: download `/Users/tranquangtrong/Desktop/ppe-checker/weights/yolo9s.pt`

- [ ] **Step 1: Create weights folder and download yolo9s.pt**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
mkdir -p weights
curl -L -o weights/yolo9s.pt https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo9s.pt
```

Expected: Download progress, final file size around ~14-15 MB.

- [ ] **Step 2: Verify file exists and has expected size**

```bash
ls -lh weights/yolo9s.pt
```

Expected: A file roughly 14-15 MB (not zero, not HTML).

- [ ] **Step 3: Smoke-test the detector on a small synthetic image**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
python -c "
import numpy as np
from detector import Detector
img = np.zeros((640, 640, 3), dtype=np.uint8)
d = Detector('weights/yolo9s.pt')
print('Loaded. Classes:', len(d.model.names))
print('Detections on blank image:', len(d.predict(img, 0.5, 0.3)))
"
```

Expected: `Loaded. Classes: 17` and `Detections on blank image: 0` (or a very small number of false positives).

- [ ] **Step 4: Verify the model's class names match `config.CLASS_NAMES`**

```bash
python -c "
from detector import Detector
from config import CLASS_NAMES
d = Detector('weights/yolo9s.pt')
mismatches = [(i, d.model.names[i], CLASS_NAMES[i]) for i in range(17) if d.model.names[i] != CLASS_NAMES[i]]
print('Mismatches:', mismatches if mismatches else 'NONE')
"
```

Expected: `Mismatches: NONE` (if mismatch occurs — class names differ slightly e.g. `face-mask` vs `face-mask-medical` — update `config.CLASS_NAMES` and `RULE_PRESETS` accordingly and re-run unit tests).

- [ ] **Step 5: Commit (no code change unless mismatch found)**

If a mismatch was found and you updated `config.py`:

```bash
git add config.py
git commit -m "fix: align CLASS_NAMES with actual weight model.names"
```

Otherwise skip this commit.

---

## Task 7: Streamlit app skeleton

**Files:**
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/app.py`

- [ ] **Step 1: Create minimal `app.py` with upload + display**

File: `/Users/tranquangtrong/Desktop/ppe-checker/app.py`

```python
"""Streamlit UI for SH17 PPE Compliance Checker."""

import io

import numpy as np
import streamlit as st
from PIL import Image

import config


st.set_page_config(page_title="SH17 PPE Compliance Checker", layout="wide")
st.title("SH17 PPE Compliance Checker")

uploaded = st.sidebar.file_uploader(
    "Upload workplace image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload a JPG or PNG image to begin. Use the sidebar.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

image_np = np.array(image)
h, w = image_np.shape[:2]

if w > config.MAX_INFERENCE_WIDTH:
    new_w = config.MAX_INFERENCE_WIDTH
    new_h = int(h * new_w / w)
    image = image.resize((new_w, new_h))
    image_np = np.array(image)
    st.info(f"Image was resized for inference: {w}x{h} -> {new_w}x{new_h}")

st.image(image_np, caption="Uploaded image", width=config.MAX_DISPLAY_WIDTH)
```

- [ ] **Step 2: Run the app**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
streamlit run app.py
```

Expected: Streamlit opens in browser. Sidebar shows file uploader. Uploading any JPG/PNG shows it in the main panel.

- [ ] **Step 3: Test with a sample image**

Open a SH17 sample image in browser via the `data/list_of_all_urls.csv` from the SH17 dataset folder — pick one URL, paste it in the browser, save it locally to `/tmp/test_ppe.jpg`, then upload through the app.

Verify: image displays without errors.

Stop the Streamlit server with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: streamlit skeleton with image upload and display"
```

---

## Task 8: Wire detection into the app

**Files:**
- Modify: `/Users/tranquangtrong/Desktop/ppe-checker/app.py`

- [ ] **Step 1: Add cached detector loader and run detection**

Replace the entire contents of `/Users/tranquangtrong/Desktop/ppe-checker/app.py` with:

```python
"""Streamlit UI for SH17 PPE Compliance Checker."""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import config
from detector import Detector


@st.cache_resource
def load_detector() -> Detector:
    return Detector(config.WEIGHTS_PATH)


def annotate(image_rgb: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes for every detection. Color by class group."""
    img = image_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if det.class_id == config.PERSON_CLASS_ID:
            color = config.GROUP_COLORS_BGR["person"]
        elif det.class_id in config.PERSON_GROUP_CLASS_IDS:
            color = config.GROUP_COLORS_BGR["head_face"]
        elif det.class_name in config.PPE_TO_GROUP:
            color = config.GROUP_COLORS_BGR["ppe_found"]
        else:
            color = config.GROUP_COLORS_BGR["other"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color[::-1], 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            img, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 1, cv2.LINE_AA,
        )
    return img


def resize_for_display(image_rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    new_w = max_width
    new_h = int(h * new_w / w)
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


st.set_page_config(page_title="SH17 PPE Compliance Checker", layout="wide")
st.title("SH17 PPE Compliance Checker")

uploaded = st.sidebar.file_uploader(
    "Upload workplace image", type=["jpg", "jpeg", "png"]
)

try:
    detector = load_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if uploaded is None:
    st.info("Upload a JPG or PNG image to begin. Use the sidebar.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

image_np = np.array(image)
h, w = image_np.shape[:2]
if w > config.MAX_INFERENCE_WIDTH:
    new_w = config.MAX_INFERENCE_WIDTH
    new_h = int(h * new_w / w)
    image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    st.info(f"Image was resized for inference: {w}x{h} -> {new_w}x{new_h}")

with st.spinner("Detecting..."):
    detections = detector.predict(
        image_np,
        conf_person=config.DEFAULT_CONF_PERSON,
        conf_ppe=config.DEFAULT_CONF_PPE,
    )

annotated = annotate(image_np, detections)
annotated_display = resize_for_display(annotated, config.MAX_DISPLAY_WIDTH)

st.image(annotated_display, caption=f"{len(detections)} detections")
```

- [ ] **Step 2: Run and verify**

```bash
streamlit run app.py
```

Upload a SH17 sample image. Expected: bounding boxes appear over detected persons / PPE. Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: wire YOLO detection and bbox annotation into Streamlit"
```

---

## Task 9: Sidebar config and compliance table

**Files:**
- Modify: `/Users/tranquangtrong/Desktop/ppe-checker/app.py`

- [ ] **Step 1: Replace `app.py` with the full version**

Replace the entire contents of `/Users/tranquangtrong/Desktop/ppe-checker/app.py` with:

```python
"""Streamlit UI for SH17 PPE Compliance Checker."""

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from compliance import check_compliance
from detector import Detector


@st.cache_resource
def load_detector() -> Detector:
    return Detector(config.WEIGHTS_PATH)


def annotate(image_rgb: np.ndarray, detections, compliance_results) -> np.ndarray:
    """Draw boxes. Persons get index labels; PPE-found vs PPE-orphan colored differently."""
    img = image_rgb.copy()
    persons_iter = iter(compliance_results)
    persons_drawn = 0
    accepted_ppe_ids = set()
    for pc in compliance_results:
        for group_dets in pc.found_ppe.values():
            for d in group_dets:
                accepted_ppe_ids.add(id(d))

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if det.class_id == config.PERSON_CLASS_ID:
            color = config.GROUP_COLORS_BGR["person"]
            label = f"Person #{persons_drawn + 1} {det.confidence:.2f}"
            persons_drawn += 1
        elif det.class_id in config.PERSON_GROUP_CLASS_IDS:
            color = config.GROUP_COLORS_BGR["head_face"]
            label = f"{det.class_name} {det.confidence:.2f}"
        elif det.class_name in config.PPE_TO_GROUP:
            if id(det) in accepted_ppe_ids:
                color = config.GROUP_COLORS_BGR["ppe_found"]
            else:
                color = config.GROUP_COLORS_BGR["ppe_orphan"]
            label = f"{det.class_name} {det.confidence:.2f}"
        else:
            color = config.GROUP_COLORS_BGR["other"]
            label = f"{det.class_name} {det.confidence:.2f}"
        rgb = color[::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
        cv2.putText(
            img, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1, cv2.LINE_AA,
        )
    return img


def resize_for_display(image_rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    new_w = max_width
    new_h = int(h * new_w / w)
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_compliance_df(compliance_results, required_groups: list[str]) -> pd.DataFrame:
    rows = []
    for pc in compliance_results:
        row = {"Person": f"#{pc.person_idx + 1}"}
        for g in required_groups:
            items = pc.found_ppe.get(g, [])
            if items:
                names = ", ".join(sorted({d.class_name for d in items}))
                row[g] = f"OK ({names})"
            else:
                row[g] = "MISSING"
        row["Status"] = "Compliant" if pc.is_compliant else "Violation"
        rows.append(row)
    return pd.DataFrame(rows)


st.set_page_config(page_title="SH17 PPE Compliance Checker", layout="wide")
st.title("SH17 PPE Compliance Checker")

st.sidebar.header("Detection Settings")
conf_person = st.sidebar.slider(
    "Person/Head confidence",
    min_value=0.05, max_value=0.95, value=config.DEFAULT_CONF_PERSON, step=0.05,
)
conf_ppe = st.sidebar.slider(
    "PPE confidence",
    min_value=0.05, max_value=0.95, value=config.DEFAULT_CONF_PPE, step=0.05,
)

st.sidebar.header("Required PPE")
req_head = st.sidebar.checkbox("Head (Helmet)", value=True)
req_body = st.sidebar.checkbox("Body (Vest/Suit)", value=True)
req_hand = st.sidebar.checkbox("Hand (Gloves)", value=False)
req_foot = st.sidebar.checkbox("Foot (Shoes)", value=False)

required_groups: list[str] = []
if req_head:
    required_groups.append("HEAD")
if req_body:
    required_groups.append("BODY")
if req_hand:
    required_groups.append("HAND")
if req_foot:
    required_groups.append("FOOT")

with st.sidebar.expander("About"):
    st.markdown(
        "**Dataset:** SH17 (8,099 images, 17 PPE-related classes; "
        "[GitHub](https://github.com/ahmadmughees/SH17dataset), CC BY-NC-SA 4.0).\n\n"
        "**Model:** YOLOv9-s pre-trained on SH17 (paper mAP50 = 65.3%).\n\n"
        "**Limitations:** Rare PPE classes (helmet, safety-vest, face-guard, "
        "safety-suit, medical-suit) have lower recall. Lower the PPE confidence "
        "threshold if items are visibly present but undetected.\n\n"
        "**Spatial association** is approximate. A helmet is attributed to a person "
        "when its bounding box is at least 70% inside the person's bounding box. "
        "Occlusion or close stacks of people may cause incorrect attribution. "
        "This app is for educational/demo purposes, not safety-critical use."
    )

uploaded = st.sidebar.file_uploader(
    "Upload workplace image", type=["jpg", "jpeg", "png"]
)

try:
    detector = load_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if uploaded is None:
    st.info("Upload a JPG or PNG image to begin. Use the sidebar.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

image_np = np.array(image)
h, w = image_np.shape[:2]
if w > config.MAX_INFERENCE_WIDTH:
    new_w = config.MAX_INFERENCE_WIDTH
    new_h = int(h * new_w / w)
    image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    st.info(f"Image was resized for inference: {w}x{h} -> {new_w}x{new_h}")

with st.spinner("Detecting..."):
    detections = detector.predict(image_np, conf_person, conf_ppe)

compliance_results = check_compliance(
    detections,
    required_groups=required_groups,
    containment_threshold=config.CONTAINMENT_THRESHOLD,
)

annotated = annotate(image_np, detections, compliance_results)
annotated_display = resize_for_display(annotated, config.MAX_DISPLAY_WIDTH)

n_persons = len(compliance_results)
n_compliant = sum(1 for pc in compliance_results if pc.is_compliant)
n_violations = n_persons - n_compliant

m1, m2, m3 = st.columns(3)
m1.metric("Total persons", n_persons)
m2.metric("Compliant", n_compliant)
m3.metric("Violations", n_violations)

col_img, col_table = st.columns([2, 1])
with col_img:
    st.image(annotated_display, caption=f"{len(detections)} total detections")
with col_table:
    if not required_groups:
        st.info("Select at least one required PPE group in the sidebar.")
    elif n_persons == 0:
        st.warning(
            "No person detected. Try lowering the Person/Head confidence threshold."
        )
    else:
        df = build_compliance_df(compliance_results, required_groups)
        st.dataframe(df, hide_index=True, use_container_width=True)
```

- [ ] **Step 2: Run the app**

```bash
streamlit run app.py
```

- [ ] **Step 3: Manual verification**

Upload a SH17 sample image. Verify:
- Sliders move
- Checkboxes toggle required groups
- Metrics show person counts
- Compliance table appears with one row per person
- Bounding boxes are color-coded (orange-ish for person, green for PPE-found, cyan for PPE-orphan)
- About expander shows description

Stop with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add sidebar config and per-person compliance table"
```

---

## Task 10: Sample images and README

**Files:**
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/samples/` (folder)
- Create: `/Users/tranquangtrong/Desktop/ppe-checker/README.md`

- [ ] **Step 1: Create sample images folder and download 3-5 images**

Pick 3-5 representative URLs from the SH17 dataset's `list_of_all_urls.csv`:

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
mkdir -p samples
curl -L -o samples/sample_01.jpg "https://images.pexels.com/photos/2469/building-construction-building-site-constructing.jpg"
curl -L -o samples/sample_02.jpg "https://images.pexels.com/photos/159306/construction-site-build-construction-work-159306.jpeg"
curl -L -o samples/sample_03.jpg "https://images.pexels.com/photos/162540/career-firefighter-relaxing-job-162540.jpeg"
```

(If a URL 404s, pick a replacement from `/Users/tranquangtrong/Desktop/SH17dataset-master/data/list_of_all_urls.csv`.)

- [ ] **Step 2: Verify image files**

```bash
ls -lh samples/
file samples/*.jpg
```

Expected: 3+ JPEG files, each at least a few hundred KB.

- [ ] **Step 3: Create `README.md`**

File: `/Users/tranquangtrong/Desktop/ppe-checker/README.md`

````markdown
# SH17 PPE Compliance Checker

A Streamlit web app that detects PPE compliance violations on workplace images.
Uses pre-trained YOLOv9-s weights from the [SH17 dataset](https://github.com/ahmadmughees/SH17dataset).

## Quick start

### 1. Install

```bash
git clone <this-repo>
cd ppe-checker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download pre-trained weights

```bash
mkdir -p weights
curl -L -o weights/yolo9s.pt https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo9s.pt
```

### 3. Run

```bash
streamlit run app.py
```

A browser window opens at `http://localhost:8501`. Use the sidebar to upload a JPG/PNG image of a workplace scene.

### 4. Run tests

```bash
pytest tests/ -v
```

## How it works

1. **YOLO inference** detects 17 SH17 classes (person, head, helmet, vest, gloves, shoes, etc.) with two separate confidence thresholds: one for person/face/head, one for PPE classes (PPE often needs a lower threshold due to class imbalance in training data).
2. **Spatial association**: for each detected `person`, PPE detections whose bounding box is at least 70% inside that person's bounding box are attributed to them.
3. **Compliance check**: for each person, check whether at least one PPE of each *required group* was found:
   - HEAD: helmet
   - BODY: safety-vest, safety-suit, or medical-suit
   - HAND: gloves
   - FOOT: shoes
4. **Output**: annotated image (color-coded boxes) + table with per-person compliance status.

## Project structure

```
ppe-checker/
├── app.py            # Streamlit UI
├── detector.py       # YOLO wrapper
├── compliance.py     # Pure logic: containment + rule check
├── config.py         # Class names, rule presets, defaults
├── weights/yolo9s.pt # Pre-trained model (download separately)
├── samples/          # Test images
├── tests/            # Unit tests for compliance logic
├── docs/             # Spec and implementation plan
├── requirements.txt
└── README.md
```

## Limitations & disclaimers

- **Pre-trained weights have lower recall on rare PPE classes** (helmet 1.2% of training data, safety-vest 0.7%, face-guard 0.2%). Lower the PPE confidence threshold if items are visibly present but undetected.
- **Spatial association is heuristic.** Overlapping people, occlusion, or unusual poses can cause incorrect attribution.
- **The model does not distinguish "worn" vs "present in scene."** A helmet held in hand close to the body could be mistakenly counted as worn.
- **Not for safety-critical use.** Built as a school project / research demo.
- **License**: SH17 dataset and derived weights are CC BY-NC-SA 4.0 — non-commercial use only.

## Acknowledgements

- Dataset: Ahmad & Rahimi, *SH17: A Dataset for Human Safety and PPE Detection in Manufacturing Industry*, Journal of Safety Science and Resilience, 2024. [Paper](https://doi.org/10.1016/j.jnlssr.2024.09.002).
- YOLO: ultralytics 8.0.38.
````

- [ ] **Step 4: Commit**

```bash
git add samples/ README.md
git commit -m "docs: add README and sample images"
```

---

## Task 11: Final end-to-end verification

**Files:** No code changes unless issues found.

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/tranquangtrong/Desktop/ppe-checker
source .venv/bin/activate
pytest tests/ -v
```

Expected: 14 passed (5 containment + 9 compliance).

- [ ] **Step 2: Run the app and execute manual smoke checklist**

```bash
streamlit run app.py
```

Verify each item:

- [ ] Sidebar shows uploader, 2 sliders, 4 checkboxes, About expander.
- [ ] No image uploaded -> info banner shown.
- [ ] Upload `samples/sample_01.jpg` -> image renders with boxes.
- [ ] Default settings (HEAD + BODY required) -> metrics row shows non-zero "Total persons".
- [ ] Compliance table has one row per detected person.
- [ ] Toggle all 4 checkboxes off -> table replaced with "Select at least one required PPE group" message.
- [ ] Lower Person/Head confidence to 0.10 -> more persons detected.
- [ ] Try `samples/sample_02.jpg` and `samples/sample_03.jpg` -> renders correctly.
- [ ] Move `weights/yolo9s.pt` aside temporarily and reload -> error message with download URL shown. Restore the file.

Stop the server with Ctrl+C.

- [ ] **Step 3: If any issues found during smoke test, fix in code, re-run tests, commit**

For each issue:
1. Fix the code.
2. Re-run `pytest tests/ -v` (must still pass).
3. Re-run the smoke check.
4. `git add <files> && git commit -m "fix: <issue>"`.

- [ ] **Step 4: Final summary commit (if anything changed)**

```bash
git log --oneline
```

Expected: at least 9 commits covering: bootstrap, config, containment, compliance, detector, app skeleton, detection wiring, full UI, README+samples. Any fix commits from Step 3.

---

## Done criteria

- [ ] All 14 unit tests pass.
- [ ] `streamlit run app.py` starts without errors.
- [ ] Manual smoke checklist (Task 11 Step 2) all green.
- [ ] README clearly documents install, weight download, and run.
- [ ] At least 3 sample images in `samples/`.
- [ ] Git log has clean, atomic commits per task.
