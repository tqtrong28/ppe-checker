from compliance import containment_ratio


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
