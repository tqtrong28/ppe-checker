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
