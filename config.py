import os
from pathlib import Path

STAINING_CONFIG = {
    "Dystrophin": {
        "slide": 1,
        "czi_dir": "IHF_Lam-Dys-Col4",
        "channel": 2,
        "ihf": True,
        "dilation": 2,
        "erosion": 8,
        "display_name": "Dystrophin",
    },
    "Collagen4": {
        "slide": 1,
        "czi_dir": "IHF_Lam-Dys-Col4",
        "channel": 3,
        "ihf": True,
        "dilation": 2,
        "erosion": 8,
        "display_name": "Col4",
    },
    "Laminin": {
        "slide": 1,
        "czi_dir": "IHF_Lam-Dys-Col4",
        "channel": 1,
        "ihf": True,
        "dilation": 2,
        "erosion": 8,
        "display_name": "Laminin",
    },
    "IgG": {
        "slide": 2,
        "czi_dir": "IHF_Lam-IgG-CD11b",
        "channel": 2,
        "ihf": True,
        "dilation": 2,
        "erosion": 8,
        "display_name": "IgG",
    },
    "CD11b": {
        "slide": 2,
        "czi_dir": "IHF_Lam-IgG-CD11b",
        "channel": 3,
        "ihf": True,
        "dilation": 2,
        "erosion": 8,
        "display_name": "CD11b",
    },
    "NADH": {
        "slide": 3,
        "czi_dir": "NADH",
        "channel": 0,
        "ihf": False,
        "dilation": 0,
        "erosion": 4,
        "display_name": "NADH",
    },
    "HE_10x": {
        "slide": 6,
        "czi_dir": "HE_10X",
        "channel": 0,
        "ihf": False,
        "dilation": 0,
        "erosion": 4,
        "display_name": "HE_10x",
    },
    "COX": {
        "slide": 7,
        "czi_dir": "COX",
        "channel": 0,
        "ihf": False,
        "dilation": 0,
        "erosion": 4,
        "display_name": "COX",
    },
    "LAMP2": {
        "slide": 8,
        "czi_dir": "IHF_LAMP2-LGALS3-SQSTM1",
        "channel": 1,
        "ihf": True,
        "dilation": 0,
        "erosion": 4,
        "display_name": "LAMP2",
    },
}

STAININGS_ORDER = [
    "Laminin", "Dystrophin", "Collagen4", "IgG", "CD11b",
    "NADH", "HE_10x", "COX", "LAMP2",
]

STAININGS_SLIDE1 = {"Laminin", "Dystrophin", "Collagen4"}

FEATURE_STATISTICS = ["mean", "std", "p10", "p25", "p50", "p75", "p90", "skew", "kurt"]

COMPARTMENTS = ["whole", "membrane", "cytoplasm"]

GROUP_MAP = {
    "WT": ["TAG01", "TAG02", "TAG03", "TAG04", "TAG05",
           "QUAG01", "QUAG02", "QUAG03", "QUAG04", "QUAG05"],
    "mdx": ["TAG21", "TAG22", "TAG23", "TAG24", "TAG25",
            "QUAG21", "QUAG22", "QUAG23", "QUAG24", "QUAG25"],
    "AAV9": ["TAG26", "TAG27", "TAG28", "TAG29",
             "QUAG26", "QUAG27", "QUAG28", "QUAG29"],
    "LICA1": ["TAG31", "TAG32", "TAG33", "TAG34", "TAG35",
              "QUAG31", "QUAG32", "QUAG33", "QUAG34", "QUAG35"],
}

GROUP_OF_SAMPLE = {}
for group, samples in GROUP_MAP.items():
    for s in samples:
        GROUP_OF_SAMPLE[s] = group


def get_group(sample_name):
    return GROUP_OF_SAMPLE.get(sample_name, "unknown")


def iter_pairs(pairs_dir):
    for d in sorted(os.listdir(pairs_dir)):
        if "___vs___" not in d:
            continue
        yield d

def extract_base_names(pair_dir_name):
    ref_part, target_part = pair_dir_name.split("___vs___")
    return ref_part, target_part
