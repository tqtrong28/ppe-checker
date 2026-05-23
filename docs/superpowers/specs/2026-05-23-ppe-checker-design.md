# SH17 PPE Compliance Checker — Design Spec

**Date:** 2026-05-23
**Status:** Approved by user (pending spec review)
**Implementation timeline:** ~1 week (school project)

---

## 1. Overview

Build a Streamlit web application that performs **PPE (Personal Protective Equipment) compliance checking** on a single uploaded image from industrial workplace scenarios. The application uses **pre-trained YOLO weights** from the SH17 dataset to detect persons and PPE items, then checks whether each person is wearing required PPE grouped by body region (Head / Body / Hand / Foot).

**Problem framing:** A "hazard" in this project is defined as **a person missing required PPE**. The system attributes violations per-person and presents both a visual annotated image and a tabular compliance summary.

---

## 2. Goals & Non-Goals

### 2.1 Goals (in scope)
- Upload a single image (JPG / JPEG / PNG)
- Detect persons + 17 SH17 classes using pre-trained YOLOv9-s
- Spatially attribute each PPE detection to a specific person via bbox containment
- Check per-person compliance against 4 configurable PPE groups
- Render annotated image with colored bounding boxes
- Render compliance table: one row per person, one column per required group
- Allow user to adjust:
  - Confidence threshold for Person/Head/Face classes
  - Confidence threshold for PPE classes (typically lower due to class imbalance)
  - Which PPE groups are required (4 checkboxes)
- Display dataset/model background and disclaimer in an "About" section

### 2.2 Non-Goals (explicitly out of scope for v1)
- Video, webcam, or batch image processing
- Model training or fine-tuning
- True "worn vs present in scene" classification (use spatial heuristic as approximation)
- Multi-page navigation, authentication, persistence
- PDF / CSV report export
- Per-class confidence sliders (only 2 sliders: Person-group vs PPE-group)
- Cloud deployment (local `streamlit run` only)

---

## 3. Dataset & Model Background

### 3.1 SH17 Dataset
- 8,099 images, 75,994 instances, 17 classes
- Source: scraped from Pexels (industrial workplace photos worldwide)
- License: **CC BY-NC-SA 4.0** — non-commercial, research/educational use only
- Image sizes range from 1920×1002 up to 8192×5462
- Average 9.38 instances per image; many small objects

### 3.2 17 Classes (per `sh17.yaml`)

The pre-trained weights use the class ID ordering from `sh17.yaml`, **not** the order printed in the README. Always reference the YAML for ID mapping.

```
0:  person          9:  gloves
1:  ear            10:  helmet
2:  ear-mufs       11:  hands
3:  face           12:  head
4:  face-guard     13:  medical-suit
5:  face-mask      14:  shoes
6:  foot           15:  safety-suit
7:  tool           16:  safety-vest
8:  glasses
```

### 3.3 Pre-trained Model
- **Default:** `yolo9s.pt` (7.2M params, paper mAP50 = 65.3%, mAP50-95 = 42.9%)
- **Alternative if needed:** `yolo8n.pt` (3.2M, faster but mAP50 = 58.0%)
- Download from: https://github.com/ahmadmughees/SH17dataset/releases/tag/v1
- Must be loaded via `ultralytics==8.0.38` (other versions may produce different results)

### 3.4 Known Limitations
- **Class imbalance:** Rare PPE classes have low instance counts:
  - helmet: 1.2% (927 instances)
  - safety-vest: 0.7% (530)
  - face-guard: 0.2% (134)
  - medical-suit: 0.2% (155)
- **Recall on rare classes is lower than aggregate mAP50 suggests.** Users should lower PPE confidence threshold to compensate.
- **No "on/off" distinction in base 17-class labels:** A helmet sitting on a table is detected as `helmet`. The spatial containment heuristic in `compliance.py` is the workaround — orphan PPE outside any Person bbox does not count toward compliance.

---

## 4. Architecture

### 4.1 Folder Structure

```
ppe-checker/
├── app.py                # Streamlit UI entry point
├── detector.py           # YOLO inference wrapper
├── compliance.py         # Spatial association + rule evaluation (pure logic)
├── config.py             # Constants: class names, rule presets, defaults, colors
├── weights/
│   └── yolo9s.pt         # downloaded manually from SH17 releases
├── samples/              # 3-5 test images sourced from Pexels URLs
├── tests/
│   └── test_compliance.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-23-ppe-checker-design.md  # this file
├── requirements.txt
└── README.md
```

### 4.2 Module Responsibilities

#### `config.py` — Single source of truth

```python
CLASS_NAMES: dict[int, str]  # 0..16 → name per sh17.yaml
PERSON_CLASS_ID: int = 0
PERSON_GROUP_CLASS_IDS: set[int] = {0, 3, 12}  # person, face, head — use Person/Head conf
PPE_CLASS_IDS: set[int]  # all others — use PPE conf

RULE_PRESETS: dict[str, list[str]] = {
    "HEAD": ["helmet"],
    "BODY": ["safety-vest", "safety-suit", "medical-suit"],
    "HAND": ["gloves"],
    "FOOT": ["shoes"],
}

DEFAULT_CONF_PERSON: float = 0.5
DEFAULT_CONF_PPE: float = 0.3
CONTAINMENT_THRESHOLD: float = 0.7

WEIGHTS_PATH: str = "weights/yolo9s.pt"
MAX_DISPLAY_WIDTH: int = 1024
MAX_INFERENCE_WIDTH: int = 4096  # auto-resize if input larger

BBOX_COLORS: dict[str, tuple[int, int, int]]  # group → BGR color
```

#### `detector.py` — YOLO wrapper

```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

class Detector:
    def __init__(self, weights_path: str): ...
    def predict(
        self,
        image: np.ndarray,
        conf_person: float,
        conf_ppe: float,
    ) -> list[Detection]:
        # 1. Run inference once with conf = min(conf_person, conf_ppe)
        # 2. Post-filter:
        #    - Keep if class_id in PERSON_GROUP_CLASS_IDS and conf >= conf_person
        #    - Else keep if conf >= conf_ppe
        # 3. Return Detection list
```

#### `compliance.py` — Pure logic (no I/O, no Streamlit)

```python
def containment_ratio(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    """Returns intersection_area(inner, outer) / area(inner), or 0.0 if inner has zero area."""

@dataclass
class PersonCompliance:
    person_idx: int                          # stable index 0..N-1
    person_bbox: tuple[int, int, int, int]
    person_conf: float
    found_ppe: dict[str, list[Detection]]    # group → list of PPE inside this person
    violations: list[str]                    # subset of required_groups with no PPE found
    @property
    def is_compliant(self) -> bool: ...

def check_compliance(
    detections: list[Detection],
    required_groups: list[str],
    containment_threshold: float = 0.7,
) -> list[PersonCompliance]:
    """
    1. Split detections into persons (class_id == 0) and PPE items (class_name in any RULE_PRESET value).
    2. Build reverse map: ppe_class_name → group.
    3. For each person:
       - Initialize found_ppe = {g: [] for g in required_groups}
       - For each PPE detection:
         - If containment_ratio(ppe.bbox, person.bbox) >= containment_threshold
         - And ppe.class_name maps to a group in required_groups
         - Append to found_ppe[group]
       - violations = [g for g in required_groups if not found_ppe[g]]
    4. Return list ordered by person_idx.
    """
```

#### `app.py` — Streamlit UI

**Sidebar:**
- App title + tagline
- File uploader: `accept=["jpg","jpeg","png"]`, single file
- **Detection Settings** section:
  - Slider "Person/Head confidence" — range 0.0–1.0, default 0.5
  - Slider "PPE confidence" — range 0.0–1.0, default 0.3
- **Required PPE** section (4 checkboxes):
  - Head (Helmet) — default ON
  - Body (Vest/Suit) — default ON
  - Hand (Gloves) — default OFF
  - Foot (Shoes) — default OFF
- **About** expander:
  - SH17 dataset summary (counts, license)
  - Model card (variant, mAP from paper)
  - Disclaimer on rare-class recall and "worn vs present" approximation

**Main panel (when image uploaded):**
1. Show "Detecting..." spinner during inference
2. Two columns:
   - **Left:** annotated image
     - Resize to `MAX_DISPLAY_WIDTH` preserving aspect
     - Draw bbox per Detection. Color rules:
       - Person bbox: blue, label "Person #N"
       - Head/Face: gray
       - PPE matching a required group, found inside a person: green
       - PPE detected but orphan (not in any person bbox): yellow
       - Other classes (tool, ear, hands, etc.): light gray, optional toggle
   - **Right:** compliance table (pandas DataFrame via `st.dataframe`)
     - Rows: Person #1, #2, ...
     - Columns: one per required group, cell = "✓ (helmet)" or "✗"
     - Final column: Status = Compliant / Violation
3. Metrics row above table:
   - `st.metric("Total persons", N)`, `st.metric("Compliant", Y)`, `st.metric("Violations", Z)`

**Main panel (no image):**
- Instructions + thumbnails of 3 sample images from `samples/`

### 4.3 Data Flow

```
[User uploads file]
       ↓
[PIL.Image.open() → np.ndarray (RGB)]
       ↓
[Auto-resize if width > MAX_INFERENCE_WIDTH (preserve aspect ratio)]
       ↓
[Detector.predict(image, conf_person, conf_ppe)]
       ↓
[List[Detection]]
       ↓
[compliance.check_compliance(detections, required_groups, threshold=0.7)]
       ↓
[List[PersonCompliance]]
       ↓
[Annotate image with cv2 (resized to MAX_DISPLAY_WIDTH)]
   +
[Build pandas DataFrame from compliance list]
       ↓
[st.image(annotated) + st.dataframe(compliance_df) + st.metric(...)]
```

---

## 5. Error Handling

| Scenario | Handling |
|---|---|
| Wrong file type | `st.error("Please upload a JPG, JPEG, or PNG image.")`; abort processing |
| File corrupt / PIL fails | Catch exception; `st.error("Could not read image: {msg}")` |
| Image width > `MAX_INFERENCE_WIDTH` | Auto-resize before inference; show `st.info("Image was resized for inference: WxH → W'xH'")` |
| `weights/yolo9s.pt` missing on startup | `st.error("Weight file not found. Download from: <SH17 releases URL>")`; instructions in README |
| No required groups selected (all checkboxes off) | `st.info("Select at least one PPE group to evaluate compliance.")`; still show detection without compliance table |
| Zero Person detections | `st.warning("No person detected. Try lowering Person/Head confidence threshold.")`; still show annotated image with any non-person detections |
| YOLO inference raises | Catch broadly; `st.error("Detection failed: {error}")`; log full traceback to console |

---

## 6. Testing Strategy

### 6.1 Unit Tests (`tests/test_compliance.py`)

Run with `pytest tests/ -v`. All tests use synthetic Detection lists — no model loading required.

**`containment_ratio` tests:**
- `test_containment_full_inside` — inner fully inside outer → returns 1.0
- `test_containment_half_overlap` — inner 50% overlaps outer → returns 0.5
- `test_containment_no_overlap` — disjoint bboxes → returns 0.0
- `test_containment_zero_area_inner` — inner has x1==x2 → returns 0.0 (no division by zero)
- `test_containment_outer_inside_inner` — outer smaller than inner → returns < 1.0 (clipped)

**`check_compliance` tests:**
- `test_one_person_fully_compliant` — 1 person + helmet+vest+gloves+shoes all inside → `violations == []`
- `test_one_person_missing_helmet` — same but no helmet → `violations == ["HEAD"]`
- `test_orphan_helmet_outside_person` — helmet bbox outside person bbox → HEAD still in violations
- `test_two_persons_one_compliant_one_not` — 2 persons, helmet only in person1 → person1 compliant, person2 has HEAD violation
- `test_body_group_satisfied_by_safety_suit` — person with safety-suit (not vest) → BODY satisfied
- `test_body_group_satisfied_by_medical_suit` — same with medical-suit → BODY satisfied
- `test_no_required_groups` — empty `required_groups` → `violations == []` for everyone
- `test_partial_containment_below_threshold` — PPE 60% inside person, threshold 0.7 → does NOT count, violation reported

### 6.2 Manual Smoke Tests
- 3-5 sample images in `samples/` chosen from SH17 Pexels URLs covering:
  - Single worker fully equipped
  - Single worker visibly missing helmet
  - Multi-person scene (2-3 people)
  - Indoor scene with poor lighting
  - Worker with tool in hand
- For each: visually verify bounding boxes are roughly in the right place and compliance table matches what is visible.

### 6.3 Acceptance Criteria
- All unit tests pass (`pytest tests/`)
- `streamlit run app.py` starts without errors
- Manual smoke tests subjectively look correct
- README contains: install steps, weight download instructions, run command, brief screenshots

---

## 7. Dependencies (`requirements.txt`)

```
streamlit>=1.30.0
ultralytics==8.0.38
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pandas>=2.0.0
pytest>=7.4.0
```

Python version: 3.10+ recommended (matches ultralytics compatibility).

---

## 8. Disclaimers (for README + UI "About")

- **License:** SH17 dataset is CC BY-NC-SA 4.0. This application uses pre-trained weights derived from that dataset. Non-commercial / educational use only.
- **Model limitations:** Pre-trained weights have lower recall on rare PPE classes (helmet, safety-vest, face-guard, medical-suit, safety-suit). Lower the PPE confidence threshold if items are visibly present but not detected.
- **Spatial association is approximate.** Occlusion, overlapping persons, or unusual poses can cause incorrect PPE-to-person attribution.
- **No "worn" detection.** The system flags PPE as "found" if its bounding box is contained inside a person's bounding box. This is a heuristic — a helmet held in hand close to the body could be mistakenly counted.
- **Not for safety-critical use.** Intended for coursework, demo, and research. Do not deploy as a production safety system.

---

## 9. Open Questions / Future Work

Out of scope for v1 but worth noting in README:

- Support multiple model variants (YOLOv9-e for higher accuracy, YOLOv8n for speed) via dropdown
- Per-class confidence sliders in an "Advanced" expander
- Visualize the containment-threshold effect with a debug overlay
- Video / webcam processing
- Fine-tune model with extended SH17 annotations that include `on`/`off` tags to address the "worn vs present" gap directly
- Export annotated image and compliance CSV
