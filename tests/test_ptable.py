"""The periodic-table layout of the element figure."""

import pytest

from envtok_mace.plots.ptable import LANTHANIDE_ROW, LANTHANIDES, LAST_Z, position


def test_every_element_has_its_own_cell():
    cells = [position(z) for z in range(1, LAST_Z + 1)]
    assert len(set(cells)) == LAST_Z
    assert all(0 <= c <= 17 for _, c in cells)


@pytest.mark.parametrize(
    "z, cell",
    [(1, (0, 0)), (2, (0, 17)), (5, (1, 12)), (21, (3, 2)), (56, (5, 1)), (72, (5, 3)), (86, (5, 17))],
)
def test_known_positions(z, cell):
    assert position(z) == cell


def test_the_lanthanides_sit_in_their_own_row():
    assert [position(z) for z in LANTHANIDES] == [(LANTHANIDE_ROW, 2 + i) for i in range(15)]


def test_heavier_elements_are_refused():
    with pytest.raises(ValueError):
        position(LAST_Z + 1)
