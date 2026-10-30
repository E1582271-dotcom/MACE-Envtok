"""Positions of the elements H to Rn in an 18-column periodic table.

Row 0 is period 1. The lanthanides La to Lu sit in their own row below the
table (row 7), shifted to start under group 3, as printed tables do.
"""

from __future__ import annotations

LAST_Z = 86  # Rn; OMat24 has no heavier element
LANTHANIDES = range(57, 72)
LANTHANIDE_ROW = 7


def position(z: int) -> tuple[int, int]:
    """(row, column) of element ``z``, both from 0."""
    if not 1 <= z <= LAST_Z:
        raise ValueError(f"no position for Z = {z}")
    if z in LANTHANIDES:
        return LANTHANIDE_ROW, 2 + z - 57
    if z <= 2:
        return 0, 0 if z == 1 else 17
    # (first Z of the period, row); s-block in columns 0-1, p-block ending in column 17
    for first, row in ((3, 1), (11, 2), (19, 3), (37, 4), (55, 5)):
        last = {1: 10, 2: 18, 3: 36, 4: 54, 5: 86}[row]
        if z <= last:
            if z - first < 2:
                return row, z - first
            return row, 17 - (last - z)
    raise AssertionError(z)
