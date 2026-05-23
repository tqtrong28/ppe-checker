from compliance import containment_ratio
from compliance import PersonCompliance, check_compliance


def make_det(class_id: int, class_name: str, bbox, conf=0.9):
    """Helper: build a Detection-like object for tests."""
    from detector import Detection
    return Detection(class_id=class_id, class_name=class_name, bbox=bbox, confidence=conf)


def test_containment_full_inside():
    inner = (10, 10, 20, 20)
    outer = (0, 0, 100, 100)
    assert containment_ratio(inner, outer) == 1.0


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
