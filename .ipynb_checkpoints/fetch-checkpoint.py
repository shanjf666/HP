import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

EXAMPLE_COUNTS: Dict[str, int] = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING: Dict[str, List[str]] = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def get_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate local RewardBench evaluations (no Beaker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing per‑experiment *.json metrics files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="CSV file path where the aggregated DataFrame will be saved.",
    )
    parser.add_argument(
        "--experiment_prefix",
        default="",
        help="Only use JSON files whose basenames start with this prefix.",
    )
    parser.add_argument(
        "--feature_counts_dir",
        default=None,
        type=Path,
        help="Directory of per‑experiment feature‑count JSONs (optional).",
    )
    parser.add_argument(
        "--experiments_file",
        default=None,
        type=Path,
        help="TXT file mapping <experiment_id>::feat1___feat2 (optional).",
    )
    parser.add_argument(
        "--gpt4_threshold_score",
        type=float,
        default=None,
        help="If provided, create a binary 'label' column (Overall > threshold).",
    )
    parser.add_argument(
        "--dataset_total_size",
        type=int,
        default=None,
        help="Original dataset size (for budget scaling when using feature_counts_dir).",
    )
    return parser.parse_args()


def main():
    args = get_args()

    logging.info("Collecting experiments from %s", args.results_dir)

    overall_df = fetch_evals_rewardbench(
        results_dir=args.results_dir,
        experiment_prefix=args.experiment_prefix,
        feature_counts_dir=args.feature_counts_dir,
        experiments_file=args.experiments_file,
        gpt4_threshold_score=args.gpt4_threshold_score,
        dataset_total_size=args.dataset_total_size,
    )

    logging.info("Saving %d rows to %s", len(overall_df), args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(args.output_path)
    logging.info("Saved!")


def fetch_evals_rewardbench(
    *,
    results_dir: Path,
    experiment_prefix: str = "",
    feature_counts_dir: Optional[Path] = None,
    experiments_file: Optional[Path] = None,
    gpt4_threshold_score: Optional[float] = None,
    dataset_total_size: Optional[int] = None,
) -> pd.DataFrame:
    """Read local JSON metrics and assemble the full score dataframe."""

    # ------------------------------------------------------------------
    # 1. Load subset scores ------------------------------------------------
    # ------------------------------------------------------------------
    metric_files = [
        p for p in results_dir.glob("*.json") if p.stem.startswith(experiment_prefix)
    ]
    if not metric_files:
        raise FileNotFoundError(
            f"No *.json files starting with '{experiment_prefix}' found in {results_dir}"
        )

    logging.info("Found %d metric files matching prefix", len(metric_files))

    subset_scores = {}
    for fp in tqdm(metric_files, desc="reading metrics"):
        with open(fp, "r") as f:
            raw = json.load(f)

        # Some evaluation scripts store subset scores under a nested key, commonly
        # called "extra_results".  Detect that and flatten automatically.
        if isinstance(raw, dict) and "extra_results" in raw and isinstance(raw["extra_results"], dict):
            metrics_dict = raw["extra_results"]
        else:
            metrics_dict = raw

        # Keep only scalar numeric entries — drop paths, model names, etc.
        numeric_metrics = {
            k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))
        }
        subset_scores[fp.stem] = numeric_metrics

    # Build DataFrame restricted to known subset columns
    df_subset_scores = pd.DataFrame(subset_scores).transpose()
    known_cols = [c for c in df_subset_scores.columns if c in EXAMPLE_COUNTS]
    if not known_cols:
        logging.error(
            "None of the recognised subset names were found. Available columns: %s",
            list(df_subset_scores.columns),
        )
    else:
        df_subset_scores = df_subset_scores[known_cols]
    # ------------------------------------------------------------------
    # 2. Compute category & overall scores ---------------------------------
    # ------------------------------------------------------------------ and drop non‑subset columns like 'model'
    df_subset_scores = (
        pd.DataFrame(subset_scores).transpose().drop(columns=["model"], errors="ignore")
    )

    # ------------------------------------------------------------------
    # 2. Compute category & overall scores ---------------------------------
    # ------------------------------------------------------------------
    logging.info("Computing category scores…")
    df_category_scores = get_category_scores(df_subset_scores).sort_values(
        by="Overall", ascending=False
    )

    # ------------------------------------------------------------------
    # 3. Merge with feature counts or mapping (optional) -------------------
    # ------------------------------------------------------------------
    if feature_counts_dir:
        logging.info("Merging with feature_counts_dir=%s", feature_counts_dir)

        df_category_scores = df_category_scores[
            df_category_scores.index.str.contains("ID")
        ]
        df_subset_scores = df_subset_scores[df_subset_scores.index.str.contains("ID")]

        # Extract uuid + budget from experiment name
        df_category_scores["uuid"] = df_category_scores.index.to_series().apply(
            lambda x: re.search(r"ID__([a-f0-9]+)__", x).group(1)
        )
        df_category_scores["budget"] = df_category_scores.index.to_series().apply(
            lambda x: int(re.search(r"SWAPS_(\d+)", x).group(1))
        )
        if dataset_total_size:
            df_category_scores["budget"] = df_category_scores["budget"].apply(
                lambda x: int(x * 7000 / dataset_total_size)
            )

        df_subset_scores["uuid"] = df_subset_scores.index.to_series().apply(
            lambda x: re.search(r"ID__([a-f0-9]+)__", x).group(1)
        )

        # Read feature‑count JSONs (one file per experiment)
        feats = []
        for feat_file in feature_counts_dir.glob("*.json"):
            uuid_match = re.search(r"ID__([a-f0-9]+)__", feat_file.stem)
            if not uuid_match:
                continue
            uuid = uuid_match.group(1)
            with open(feat_file) as f:
                feat_dict = json.load(f)
            df_feat = (
                pd.Series(feat_dict, name=uuid)
                .to_frame()
                .transpose()
                .reset_index()
                .rename(columns={"index": "uuid"})
            )
            feats.append(df_feat)
        df_feats = pd.concat(feats, ignore_index=True)

        df_scores = df_category_scores.merge(df_subset_scores, on="uuid", how="left")
        overall_df = df_scores.merge(df_feats, on="uuid", how="left").dropna()

    elif experiments_file:
        logging.info("Merging with experiments_file=%s", experiments_file)

        df_feats = get_features(
            df_category_scores.reset_index().rename(columns={"index": "experiment"}),
            col_name="experiment",
            experiments_file=experiments_file,
        )

        def extract_hash(s: str):
            m = re.search(r"FEATS_(.*?)_SWAPS", s)
            return m.group(1) if m else None

        def extract_swaps(s: str):
            m = re.search(r"SWAPS_(\d+)", s)
            return int(m.group(1)) if m else None

        for df_ in (df_feats, df_category_scores, df_subset_scores):
            df_["hash"] = df_.index.to_series().apply(extract_hash)
        df_feats["num_swaps"] = df_feats.index.to_series().apply(extract_swaps)

        overall_df = (
            pd.merge(df_feats, df_category_scores, on="hash", how="inner")
            .reset_index()
            .merge(df_subset_scores, on="hash", how="inner")
            .set_index("hash")
        )
    else:
        overall_df = df_category_scores.merge(
            df_subset_scores, left_index=True, right_index=True
        )

    # ------------------------------------------------------------------
    # 4. Add binary label if requested -------------------------------------
    # ------------------------------------------------------------------
    if gpt4_threshold_score is not None:
        logging.info("Applying GPT‑4 threshold: %.4f", gpt4_threshold_score)
        overall_df["label"] = (overall_df["Overall"] > gpt4_threshold_score).astype(int)

    # ------------------------------------------------------------------
    # 5. Re‑order columns & return -----------------------------------------
    # ------------------------------------------------------------------
    meta_cols = [c for c in ("model_type", "chat_template") if c in overall_df.columns]
    other_cols = [c for c in overall_df.columns if c not in meta_cols]
    overall_df = overall_df[meta_cols + other_cols]

    overall_df = (
        overall_df.sort_values(by="Overall", ascending=False)
        .loc[~overall_df.index.duplicated(keep="first")]
    )

    return overall_df


def get_category_scores(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Weighted category + overall averages, tolerant to missing columns."""

    category_scores = {}
    missing_any = False
    for category, subsets in SUBSET_MAPPING.items():
        present = [s for s in subsets if s in df_subset.columns]
        if not present:
            logging.warning(
                "Category '%s' skipped: none of its subset columns found in metrics.",
                category,
            )
            missing_any = True
            continue
        weights = {k: v for k, v in EXAMPLE_COUNTS.items() if k in present}
        category_scores[category] = (
            df_subset[present] * pd.Series(weights)
        ).sum(axis=1) / sum(weights.values())

    if not category_scores:
        raise ValueError(
            "None of the expected subset names were found in the metrics files. "
            "Double‑check the JSON keys or update EXAMPLE_COUNTS & SUBSET_MAPPING to match them."
        )

    if missing_any:
        logging.warning(
            "Some categories were skipped due to missing columns. Update SUBSET_MAPPING "
            "or ensure your metrics JSON files include the expected subset names."
        )

    df_category = pd.DataFrame(category_scores)
    if not df_category.empty:
        df_category["Overall"] = df_category.mean(axis=1)
    return df_category


def get_features(
    df: pd.DataFrame,
    col_name: str,
    experiments_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Construct a binary feature matrix from experiment names or mapping file."""

    experiment_to_feats: Dict[str, List[str]] = {}

    if experiments_file is None:
        logging.info("Deriving features from experiment names…")
        for exp in df[col_name]:
            experiment_to_feats[exp] = exp.split("FEATS_")[-1].split("___")
    else:
        logging.info("Reading features from %s", experiments_file)
        with open(experiments_file) as f:
            for line in f.read().splitlines():
                exp_id, features = line.split("::")
                experiment_to_feats[exp_id] = [f.replace("-", "=") for f in features.split("___")]

    unique_feats = sorted({f for feats in experiment_to_feats.values() for f in feats})
    df_feats = pd.DataFrame(
        [
            {feat: int(feat in feats) for feat in unique_feats}
            for feats in experiment_to_feats.values()
        ],
        index=list(experiment_to_feats.keys()),
    )
    return df_feats


if __name__ == "__main__":
    main()
