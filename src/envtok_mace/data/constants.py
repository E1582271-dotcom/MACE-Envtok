"""Element table and reference energies shared by data, model and evaluation.

The model covers the 68 elements left in OMat24 after removing lanthanides
(Z = 57-71) and actinides (Z >= 89), the same table the paper uses.
"""

# Atomic numbers covered by the model, in the order used for one-hot encoding.
Z_TABLE: tuple[int, ...] = tuple(range(1, 57)) + tuple(range(72, 84))

# Structures containing any of these are dropped from train, val and test sets.
RARE_EARTH_Z: frozenset[int] = frozenset(range(57, 72)) | frozenset(range(89, 104))

# OMat24 PBE isolated-atom energies (eV), restricted to Z_TABLE. Taken from the
# authors' 06_data_preparation/preprocess.yaml; used as fixed per-element offsets.
E0S: dict[int, float] = {
    1: -1.11734008,
    2: 0.00096759,
    3: -0.29754725,
    4: -0.01781697,
    5: -0.26885011,
    6: -1.26173507,
    7: -3.12438806,
    8: -1.54838784,
    9: -0.51882044,
    10: -0.01241601,
    11: -0.22883163,
    12: -0.00951015,
    13: -0.21630193,
    14: -0.8263903,
    15: -1.88816619,
    16: -0.89160769,
    17: -0.25828273,
    18: -0.04925973,
    19: -0.22697913,
    20: -0.0927795,
    21: -2.11396364,
    22: -2.50054871,
    23: -3.70477179,
    24: -5.60261985,
    25: -5.32541181,
    26: -3.52004933,
    27: -1.93555024,
    28: -0.9351969,
    29: -0.60025846,
    30: -0.1651332,
    31: -0.32990651,
    32: -0.77971828,
    33: -1.68367812,
    34: -0.76941032,
    35: -0.22213843,
    36: -0.0335879,
    37: -0.1881724,
    38: -0.06826294,
    39: -2.17084228,
    40: -2.28579303,
    41: -3.13429753,
    42: -4.60211419,
    43: -3.45201492,
    44: -2.38073513,
    45: -1.46855515,
    46: -1.4773126,
    47: -0.33954585,
    48: -0.16843877,
    49: -0.35470981,
    50: -0.83642657,
    51: -1.41101987,
    52: -0.65740879,
    53: -0.18964571,
    54: -0.00857582,
    55: -0.13771876,
    56: -0.03457659,
    72: -3.52770927,
    73: -3.54492209,
    74: -4.65658356,
    75: -4.70108713,
    76: -2.88257209,
    77: -1.46779304,
    78: -0.50269936,
    79: -0.28801193,
    80: -0.12454674,
    81: -0.31737194,
    82: -0.77644932,
    83: -1.32627283,
}

# OMat24 training sub-datasets. Order matters when matching a sid to its source:
# longer names first, so "rattled-1000-subsampled" is not read as "rattled-1000".
OMAT24_SUBSETS: tuple[str, ...] = (
    "rattled-1000-subsampled",
    "rattled-500-subsampled",
    "rattled-300-subsampled",
    "rattled-1000",
    "rattled-500",
    "rattled-300",
    "rattled-relax",
    "aimd-from-PBE-1000-npt",
    "aimd-from-PBE-1000-nvt",
    "aimd-from-PBE-3000-npt",
    "aimd-from-PBE-3000-nvt",
)

assert len(Z_TABLE) == 68
assert set(E0S) == set(Z_TABLE)
assert not RARE_EARTH_Z & set(Z_TABLE)


def source_of_sid(sid: str) -> str:
    """Return the OMat24 sub-dataset a structure id belongs to."""
    for name in OMAT24_SUBSETS:
        if f"_{name}_" in sid or sid.endswith(f"_{name}"):
            return name
    raise ValueError(f"cannot tell the OMat24 sub-dataset of sid {sid!r}")
