import pandas as pd
import numpy as np
import pickle
import os
import re

from .config import (
    STAININGS_ORDER, STAININGS_SLIDE1, FEATURE_STATISTICS, COMPARTMENTS,
    extract_base_names,
)


def load_paired_labels(pair_dir_path):
    pkl_path = os.path.join(pair_dir_path, "paired_labels.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        pairs = pickle.load(f)
    ref_labels = [int(p[0]) for p in pairs]
    target_labels = [int(p[1]) for p in pairs]
    return pd.DataFrame({"ref_label": ref_labels, "target_label": target_labels})


def infer_target_staining_from_basename(target_base_name, stainings_config):
    target_upper = target_base_name.upper()
    for sname, scfg in stainings_config.items():
        czi_dir = scfg["czi_dir"].upper().replace("_", "")
        if czi_dir in target_upper:
            return sname
        if "HE" in target_upper and sname == "HE_10x":
            return sname
    return None


def compile_sample(
    sample_name,
    stainings_config,
    stainings_order,
    pair_dirs_base,
    quant_dirs_base,
    output_dir,
    max_missing_frac=0.2,
    slide1_stainings=None,
):
    if slide1_stainings is None:
        slide1_stainings = STAININGS_SLIDE1
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{sample_name}_profile.csv.gz")
    if os.path.exists(out_path):
        print(f"  Already compiled: {sample_name}")
        return pd.read_csv(out_path, compression="gzip")
    all_data = {}
    missing_counts = {}
    missing_details = {}
    pair_dir_names = sorted(os.listdir(pair_dirs_base))
    pair_dirs_for_sample = [d for d in pair_dir_names if sample_name.upper() in d.upper()]
    if len(pair_dirs_for_sample) == 0:
        print(f"  WARNING: No pair dirs for {sample_name}")
        return None
    used_stainings = set()
    for pair_dir_name in pair_dirs_for_sample:
        ref_part, target_part = extract_base_names(pair_dir_name)
        target_staining = infer_target_staining_from_basename(target_part, stainings_config)
        if target_staining is None:
            continue
        if target_staining in used_stainings:
            continue
        used_stainings.add(target_staining)
        pair_dir_path = os.path.join(pair_dirs_base, pair_dir_name)
        mapping_df = load_paired_labels(pair_dir_path)
        if mapping_df is None:
            print(f"  No paired_labels for {pair_dir_name}")
            continue
        quant_dir = quant_dirs_base
        quant_path = os.path.join(quant_dir, f"{sample_name}_{target_staining}_quant.csv")
        if not os.path.exists(quant_path):
            quant_path_alt = os.path.join(
                quant_dir, f"{sample_name}_{target_staining}_quant.csv"
            )
            if not os.path.exists(quant_path_alt):
                print(f"  No quant CSV for {target_staining}")
                continue
            quant_path = quant_path_alt
        df_quant = pd.read_csv(quant_path)
        label_col = "fiber_label"
        if label_col not in df_quant.columns:
            print(f"  No fiber_label in quant CSV for {target_staining}")
            continue
        label_to_feats = df_quant.set_index(label_col).to_dict("index")
        for _, row in mapping_df.iterrows():
            ref_lbl = row["ref_label"]
            tgt_lbl = row["target_label"]
            if ref_lbl not in all_data:
                all_data[ref_lbl] = {"ref_label": ref_lbl}
                missing_counts[ref_lbl] = 0
                missing_details[ref_lbl] = {}
            if tgt_lbl in label_to_feats:
                feats = label_to_feats[tgt_lbl]
                for key, val in feats.items():
                    col_name = f"{target_staining}_{key}"
                    all_data[ref_lbl][col_name] = val
            else:
                missing_counts[ref_lbl] = missing_counts.get(ref_lbl, 0) + 1
                missing_details[ref_lbl][target_staining] = True
    for sname in slide1_stainings:
        if sname in used_stainings:
            continue
        if sname not in stainings_config:
            continue
        used_stainings.add(sname)
        quant_dir = quant_dirs_base
        quant_path = os.path.join(quant_dir, f"{sample_name}_{sname}_quant.csv")
        if not os.path.exists(quant_path):
            print(f"  No quant CSV for slide1 staining {sname}")
            continue
        df_quant = pd.read_csv(quant_path)
        label_col = "fiber_label"
        if label_col not in df_quant.columns:
            continue
        ref_labels_in_data = set(all_data.keys())
        for _, row in df_quant.iterrows():
            lbl = row[label_col]
            if lbl not in ref_labels_in_data:
                all_data[lbl] = {"ref_label": lbl}
                missing_counts[lbl] = 0
                missing_details[lbl] = {}
            feats = row.to_dict()
            del feats[label_col]
            for key, val in feats.items():
                col_name = f"{sname}_{key}"
                all_data[lbl][col_name] = val
    df = pd.DataFrame.from_dict(all_data, orient="index")
    if "ref_label" not in df.columns:
        return None
    df = df.sort_values("ref_label").reset_index(drop=True)
    n_stainings = len([s for s in stainings_order if s != "Myh427"])
    expected = n_stainings
    n_missing_col = []
    for _, row in df.iterrows():
        count = 0
        for s in stainings_order:
            if s in slide1_stainings or s in used_stainings:
                col = f"{s}_area"
                if col not in row or pd.isna(row[col]):
                    count += 1
        n_missing_col.append(count)
    df["n_stainings_missing"] = n_missing_col
    df["frac_missing"] = df["n_stainings_missing"] / expected
    n_before = len(df)
    df = df[df["frac_missing"] <= max_missing_frac].copy()
    n_after = len(df)
    print(f"  {sample_name}: {n_before} fibers → {n_after} after missing filter (≤{max_missing_frac:.0%})")
    median_vals = {}
    for col in df.columns:
        if col in ["ref_label", "n_stainings_missing", "frac_missing"]:
            continue
        if col.endswith("skew") or col.endswith("kurt"):
            continue
        median_vals[col] = df[col].median()
    for col in df.columns:
        if col in ["ref_label", "n_stainings_missing", "frac_missing"]:
            continue
        wasnull = df[col].isna()
        if wasnull.any():
            impute_val = median_vals.get(col, 0)
            df[col] = df[col].fillna(impute_val)
            impute_col = f"{col}_was_imputed"
            df[impute_col] = wasnull.astype(int)
    df.to_csv(out_path, index=False, compression="gzip")
    print(f"  Saved compiled profile: {out_path}")
    return df
