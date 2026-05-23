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
- YOLO: ultralytics >= 8.4.
