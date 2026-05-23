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
