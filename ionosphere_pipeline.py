from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import re
import warnings
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from savgol_target_filter import SAVGOL_DIAGNOSTIC_COLUMNS, apply_leak_safe_savgol_targets
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

warnings.filterwarnings("ignore")

PLOTLY_TEMPLATE = "plotly_white"
TARGET_LABELS = {
    "foF2": "foF2",
    "MUFD": "MUFD",
}
TARGET_TITLES = {
    "foF2": "Критическая частота foF2, МГц — 2025",
    "MUFD": "МПЧ (MUFD), МГц — 2025",
}
SEASON_COLORS = {
    "Зима 24/25": "rgba(100,149,237,0.18)",
    "Весна 2025": "rgba(60,179,113,0.18)",
    "Лето 2025": "rgba(218,165,32,0.22)",
    "Осень 2025": "rgba(210,105,30,0.18)",
}
SEASON_BANDS = [
    ("Зима 24/25", "2024-12-01", "2025-02-28", "rgba(100,149,237,0.15)"),
    ("Весна 2025", "2025-03-01", "2025-05-31", "rgba(60,179,113,0.15)"),
    ("Лето 2025", "2025-06-01", "2025-08-31", "rgba(218,165,32,0.18)"),
    ("Осень 2025", "2025-09-01", "2025-11-30", "rgba(210,105,30,0.15)"),
]
MODEL_COLORS = {
    "RandomForest": "#185FA5",
    "XGBoost": "#E8593C",
    "ElasticNet": "#1D9E75",
    "IRI": "#FF8C00",
    "Аналит.": "#9B59B6",
}
PATTERN_SHAPES = {
    "RandomForest": "",
    "XGBoost": "/",
    "ElasticNet": "x",
    "IRI": ".",
    "Аналит.": "\\",
}
MODEL_ORDER = ["RandomForest", "XGBoost", "ElasticNet", "IRI", "Аналит."]
FEATURE_COLORS = {
    "TEC": "#1565C0",
    "hour_cos": "#C62828",
    "foEs": "#6D4C41",
    "hmF2": "#6A1B9A",
    "foE": "#E65100",
    "hour_sin": "#2E7D32",
    "B0": "#00838F",
    "IRI_foF2_pred": "#F44336",
    "IRI_M3000_pred": "#FF7043",
}
LINE_STYLES = {"ML": "solid", "IRI": "dash", "Аналит.": "dot"}
LINE_WIDTHS = {"ML": 2.5, "IRI": 2.0, "Аналит.": 2.0}
SHAP_RESULT_COLUMNS = ["Date", "Target", "Model", "Feature", "Pct", "Pct_s"]
OPTUNA_SHAP_RESULT_COLUMNS = [
    "TrainDays",
    "ValDays",
    "TestH",
    "Date",
    "Season",
    "Target",
    "Model",
    "Feature",
    "MeanAbsShap",
    "Pct",
    "NBackground",
    "NExplain",
    "Pct_s",
]


@dataclass(frozen=True)
class IRIConfig:
    """Parameters that control the physical IRI baseline calculation.

    Attributes:
        lat_station: Station geographic latitude in decimal degrees.
        lon_station: Station geographic longitude in decimal degrees.
        alt_km: Station altitude in kilometers above mean sea level.
        f107_val: F10.7 solar radio flux used as a fallback when date-specific
            flux data is unavailable.
    """

    lat_station: float = 35.025
    lon_station: float = 33.157
    alt_km: float = 300.0
    f107_val: float = 150.0


@dataclass(frozen=True)
class PipelineConfig:
    """Core settings for data preparation, modeling, and diagnostics.

    Attributes:
        dataset_step: Resampling step for the dataset frame. When omitted, the
            step is inferred from the source dataset timestamps.
        daily_approximate_input: Whether to approximate sparse intraday source
            values onto the dataset step before daily aggregation.
        savgol_target_columns: Target columns that should be denoised with a
            leak-safe trailing Savitzky-Golay filter before target alignment.
        savgol_polyorder: Polynomial order for Savitzky-Golay smoothing.
        use_filtered_target_labels: Whether ML/diagnostic target labels and
            aim-derived ML feature columns should be built from
            Savitzky-Golay-smoothed targets when available.
        ml_date_start: Inclusive lower bound for evaluation dates used by
            walk-forward ML/Optuna runs and aligned diagnostics.
        ml_date_end: Inclusive upper bound for evaluation dates used by
            walk-forward ML/Optuna runs and aligned diagnostics.
        forecast_h: Forecast horizon in hours for future target alignment.
        window_list: Training window lengths, in days, used for walk-forward runs.
        plot_train_days: Default training window highlighted in plots.
        verbose: Verbosity level for pipeline logging.
        model_n_jobs: Number of parallel jobs used inside tree-based estimators
            and SHAP surrogate forests. Set this to 1 when running multiple
            stations in parallel to avoid CPU oversubscription.
        targets: Target columns to predict.
        min_feature_coverage: Minimum non-null share required for a feature.
        min_train_rows: Minimum number of rows required to train a model.
        min_train_target_rows: Minimum rows with valid target labels in training.
        min_eval_rows: Minimum rows required to score an evaluation window.
        min_cs: Minimum confidence score accepted from the GIRO dataset.
        tau_km: Effective ionospheric slab thickness for the analytic baseline.
        shap_smooth_days: Rolling window used to smooth SHAP diagnostics.
        shap_sample: Maximum sample size for SHAP foreground rows.
        shap_background: Background sample size for SHAP explainers.
        shap_kernel_samples: Number of samples for kernel-based SHAP estimation.
        shap_ml_features: ML feature subset used in SHAP calculations.
        seasons: Named seasonal date ranges used for grouped evaluation.
    """

    dataset_step: str | None = None
    daily_approximate_input: bool = True
    savgol_target_columns: tuple[str, ...] = ()
    savgol_polyorder: int = 2
    use_filtered_target_labels: bool = True
    ml_date_start: date | datetime | str | pd.Timestamp | None = None
    ml_date_end: date | datetime | str | pd.Timestamp | None = None
    forecast_h: int = 24
    window_list: tuple[int, ...] = (7, 14, 21, 30)
    plot_train_days: int = 21
    verbose: int = 1
    model_n_jobs: int = -1
    targets: tuple[str, ...] = ("foF2", "MUFD")
    min_feature_coverage: float = 0.30
    min_train_rows: int = 48
    min_train_target_rows: int = 24
    min_eval_rows: int = 2
    min_cs: int = 50
    tau_km: float = 250.0
    shap_smooth_days: int = 14
    shap_sample: int = 150
    shap_background: int = 80
    shap_kernel_samples: int = 100
    shap_ml_features: tuple[str, ...] = (
        "TEC",
        "hour_cos",
        "foEs",
        "hmF2",
        "foE",
        "hour_sin",
        "B0",
        "foF2_state",
        "MUFD_state",
    )
    seasons: dict[str, tuple[datetime, datetime]] = field(
        default_factory=lambda: {
            "Зима 24/25": (datetime(2024, 12, 1), datetime(2025, 2, 28)),
            "Весна 2025": (datetime(2025, 3, 1), datetime(2025, 5, 31)),
            "Лето 2025": (datetime(2025, 6, 1), datetime(2025, 8, 31)),
            "Осень 2025": (datetime(2025, 9, 1), datetime(2025, 11, 30)),
        }
    )


@dataclass(frozen=True)
class OptunaConfig:
    """Hyperparameter search settings for the Optuna-based workflow.

    Attributes:
        n_trials: Number of Optuna trials per model and target.
        val_days: Optional validation window length in days.
        test_h: Test window length in hours for each optimization step.
        metric: Optimization metric name.
        models: Model names included in the search.
        random_state: Random seed for reproducible optimization.
        train_final_on_train_val: Whether to refit the final model on the
            combined train and validation split.
    """

    n_trials: int = 15
    val_days: int | None = None
    test_h: int = 24
    metric: str = "MAE"
    models: tuple[str, ...] = ("RandomForest", "XGBoost", "ElasticNet")
    random_state: int = 42
    train_final_on_train_val: bool = True


DEFAULT_CONFIG = PipelineConfig()
DEFAULT_IRI_CONFIG = IRIConfig()
DEFAULT_OPTUNA_CONFIG = OptunaConfig()

DEFAULT_HTTP_TIMEOUT_SECONDS = 20
SPACEWEATHER_F107_URL = "https://spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_ELEVATION_NEIGHBOR_STEP_DEG = 0.05
OPEN_ELEVATION_SEA_LEVEL_THRESHOLD_M = 5.0
DEFAULT_DATASET_STEP = pd.Timedelta(hours=1)
DATASET_STEP_SNAP_GRID = pd.Timedelta(seconds=30)
DATASET_STEP_SNAP_TOLERANCE = pd.Timedelta(seconds=2)
MIN_REASONABLE_DATASET_STEP = pd.Timedelta(minutes=1)

COLUMN_RENAME = {
    "minEs": "fminEs",
    "halfNm": "zhalfNm",
    "caleF2": "scaleF2",
    "peEs": "TypeEs",
}
FREQ_BOUNDS = (0.5, 30.0)
MUFD_BOUNDS = (0.5, 80.0)
HEIGHT_BOUNDS = (50.0, 700.0)
MUFD_RESTORE_MIN_GAP = pd.Timedelta(hours=4)
NON_FEATURE_COLUMNS = {
    "Time",
    "hour",
    "month",
    "doy",
    "fxI",
    "foF2p",
    "MD",
    "TypeEs",
    "foF2",
    "MUFD",
    "foF2_savgol",
    "MUFD_savgol",
    "foF2_target",
    "MUFD_target",
    "IRI_foF2",
    "IRI_MUFD",
    "IRI_hmF2",
    "IRI_M3000",
    "IRI_foF2_pred",
    "IRI_MUFD_pred",
    "IRI_M3000_pred",
    "anal_foF2",
    "anal_M3000",
    "anal_MUFD",
    "anal_foF2_pred",
    "anal_MUFD_pred",
}

DATASET_FILENAME_RE = re.compile(r"^(?P<stem>.+)_(?P<year>\d{4})\.txt$")


@dataclass(frozen=True)
class DatasetFile:
    station_name: str
    station_code: str
    year: int
    path: Path


def parse_giro_2025(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()

    header = next(line.lstrip("#").strip() for line in lines if line.strip().startswith("#Time"))
    tokens = header.split()
    columns = ["Time", "CS"]
    token_index = 2
    while token_index < len(tokens):
        column = COLUMN_RENAME.get(tokens[token_index], tokens[token_index])
        columns += [column, f"{column}_QD"]
        token_index += 2

    rows: list[dict[str, object]] = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        row: dict[str, object] = {"Time": parts[0], "CS": parts[1]}
        part_offset = 0
        column_index = 2
        while column_index < len(columns) - 1 and part_offset < len(parts) - 1:
            value = parts[2 + part_offset]
            row[columns[column_index]] = np.nan if value == "---" else pd.to_numeric(value, errors="coerce")
            column_index += 2
            part_offset += 2
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["Time"] = pd.to_datetime(frame["Time"], utc=True, errors="coerce")
    frame["CS"] = pd.to_numeric(frame["CS"], errors="coerce")
    return frame.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)


def parse_dataset_header_metadata(path: str | Path) -> dict[str, object]:
    path = Path(path)
    metadata: dict[str, object] = {
        "source_path": str(path),
        "station_name": None,
        "station_code": None,
        "latitude": None,
        "longitude": None,
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# Location:"):
                match = re.search(
                    r"GEO\s+([0-9.]+)([NS])\s+([0-9.]+)([EW]),\s+URSI-Code\s+([A-Z0-9_]{4,6})\s+(.+)$",
                    line.strip(),
                )
                if match:
                    lat = float(match.group(1))
                    if match.group(2) == "S":
                        lat *= -1
                    lon = float(match.group(3))
                    if match.group(4) == "W":
                        lon *= -1
                    metadata.update(
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "station_code": match.group(5).strip(),
                            "station_name": match.group(6).strip(),
                        }
                    )
                break
    return metadata


def fetch_remote_text(
    url: str,
    *,
    timeout: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
    headers: dict[str, str] | None = None,
) -> str:
    request_headers = {"User-Agent": "Mozilla/5.0"}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_f107_adjusted_flux_records(raw_table: str) -> dict[date, float]:
    records: dict[date, float] = {}
    for raw_line in raw_table.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("fluxdate") or set(line) == {"-"}:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            flux_date = datetime.strptime(parts[0], "%Y%m%d").date()
            records[flux_date] = float(parts[5])
        except ValueError:
            continue
    if not records:
        raise ValueError("Could not parse any adjusted F10.7 values from the space weather table.")
    return records


@lru_cache(maxsize=1)
def fetch_f107_adjusted_flux_records() -> dict[date, float]:
    raw_table = fetch_remote_text(
        SPACEWEATHER_F107_URL,
        timeout=60,
    )
    return parse_f107_adjusted_flux_records(raw_table)


def resolve_f107_adjusted_flux_for_date(target_date: date, fallback: float) -> float:
    try:
        records = fetch_f107_adjusted_flux_records()
    except Exception:
        return fallback
    if target_date in records:
        return records[target_date]

    month_matches = [
        (record_date, flux_value)
        for record_date, flux_value in records.items()
        if record_date.year == target_date.year and record_date.month == target_date.month
    ]
    if month_matches:
        month_matches.sort(key=lambda item: item[0])
        return month_matches[-1][1]

    historical_matches = [
        (record_date, flux_value) for record_date, flux_value in records.items() if record_date <= target_date
    ]
    if historical_matches:
        historical_matches.sort(key=lambda item: item[0])
        return historical_matches[-1][1]
    return fallback


@lru_cache(maxsize=128)
def fetch_station_altitude_km(latitude: float, longitude: float) -> float:
    locations = [
        (latitude, longitude),
        (latitude + OPEN_ELEVATION_NEIGHBOR_STEP_DEG, longitude),
        (latitude - OPEN_ELEVATION_NEIGHBOR_STEP_DEG, longitude),
        (latitude, longitude + OPEN_ELEVATION_NEIGHBOR_STEP_DEG),
        (latitude, longitude - OPEN_ELEVATION_NEIGHBOR_STEP_DEG),
        (latitude + OPEN_ELEVATION_NEIGHBOR_STEP_DEG, longitude + OPEN_ELEVATION_NEIGHBOR_STEP_DEG),
        (latitude + OPEN_ELEVATION_NEIGHBOR_STEP_DEG, longitude - OPEN_ELEVATION_NEIGHBOR_STEP_DEG),
        (latitude - OPEN_ELEVATION_NEIGHBOR_STEP_DEG, longitude + OPEN_ELEVATION_NEIGHBOR_STEP_DEG),
        (latitude - OPEN_ELEVATION_NEIGHBOR_STEP_DEG, longitude - OPEN_ELEVATION_NEIGHBOR_STEP_DEG),
    ]
    query = urlencode({"locations": "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in locations)})
    payload = json.loads(fetch_remote_text(f"{OPEN_ELEVATION_URL}?{query}"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Open-Elevation response does not contain any results.")
    elevation_m = results[0].get("elevation")
    if elevation_m is None:
        raise ValueError("Open-Elevation response does not contain elevation.")
    exact_elevation_m = float(elevation_m)
    if exact_elevation_m > OPEN_ELEVATION_SEA_LEVEL_THRESHOLD_M:
        return exact_elevation_m / 1000.0

    neighbor_elevations_m = [
        float(item["elevation"])
        for item in results[1:]
        if item.get("elevation") is not None and float(item["elevation"]) > OPEN_ELEVATION_SEA_LEVEL_THRESHOLD_M
    ]
    if neighbor_elevations_m:
        return max(neighbor_elevations_m) / 1000.0
    return exact_elevation_m / 1000.0


def parse_dataset_filename(path: str | Path) -> DatasetFile | None:
    path = Path(path)
    match = DATASET_FILENAME_RE.match(path.name)
    if not match:
        return None
    stem = match.group("stem").strip()
    year = int(match.group("year"))

    code_match = re.match(r"^(?P<name>.+)_(?P<code>[A-Z0-9_]{4,6})$", stem)
    if code_match:
        station_name = code_match.group("name").replace("_", " ").strip()
        station_code = code_match.group("code").strip().upper()
    else:
        metadata = parse_dataset_header_metadata(path)
        station_name = str(metadata.get("station_name") or stem.replace("_", " ").strip())
        station_code = str(metadata.get("station_code") or "").strip().upper()
        if not station_code:
            return None

    return DatasetFile(
        station_name=station_name,
        station_code=station_code,
        year=year,
        path=path,
    )


def discover_dataset_files(base_dir: str | Path = ".", datasets_dir: str = "datasets") -> list[DatasetFile]:
    root = Path(base_dir) / datasets_dir
    if not root.exists():
        raise FileNotFoundError(f"Datasets folder does not exist: {root}")
    dataset_files = []
    for path in sorted(root.glob("*.txt")):
        parsed = parse_dataset_filename(path)
        if parsed is not None:
            dataset_files.append(parsed)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset .txt files with parseable station metadata were found in {root}")
    return dataset_files


def list_available_stations(base_dir: str | Path = ".", datasets_dir: str = "datasets", verbose: int = 0) -> pd.DataFrame:
    rows = []
    for dataset_file in discover_dataset_files(base_dir=base_dir, datasets_dir=datasets_dir):
        verbose_print(
            f"Found dataset file: {dataset_file.path} (Station: {dataset_file.station_name}, Year: {dataset_file.year})",
            verbose=verbose,
            level=1,
        )
        rows.append(
            {
                "station_code": dataset_file.station_code,
                "station_name": dataset_file.station_name,
                "year": dataset_file.year,
                "path": str(dataset_file.path),
            }
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["station_code", "station_name"])["year"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "year_min", "max": "year_max", "count": "files"})
    )
    return summary.sort_values(["station_code", "station_name"]).reset_index(drop=True)


def resolve_dataset_source_paths(
    base_dir: str | Path = ".",
    datasets_dir: str = "datasets",
    station_code: str | None = None,
) -> tuple[dict[int, Path], dict[str, object]]:
    dataset_files = discover_dataset_files(base_dir=base_dir, datasets_dir=datasets_dir)
    normalized_query = station_code.strip().upper() if station_code is not None else None

    if normalized_query is None:
        available_codes = sorted({dataset_file.station_code for dataset_file in dataset_files})
        if len(available_codes) != 1:
            raise ValueError(
                "Multiple stations are available in the datasets folder. "
                f"Pass station_code explicitly. Available: {', '.join(available_codes)}"
            )
        normalized_query = available_codes[0]

    selected_files = [dataset_file for dataset_file in dataset_files if dataset_file.station_code == normalized_query]
    if not selected_files:
        available_codes = sorted({dataset_file.station_code for dataset_file in dataset_files})
        raise ValueError(
            f"Station {normalized_query!r} was not found in datasets. Available: {', '.join(available_codes)}"
        )

    source_paths = {dataset_file.year: dataset_file.path for dataset_file in selected_files}
    sample_metadata = parse_dataset_header_metadata(selected_files[0].path)
    sample_metadata.setdefault("station_code", normalized_query)
    sample_metadata.setdefault("station_name", selected_files[0].station_name)
    sample_metadata["available_years"] = sorted(source_paths)
    sample_metadata["datasets_dir"] = str((Path(base_dir) / datasets_dir).resolve())
    return source_paths, sample_metadata


def load_giro_dataset(
    paths: dict[int, str | Path],
    min_cs: int = DEFAULT_CONFIG.min_cs,
    verbose: int = 0,
) -> pd.DataFrame:
    frames = []
    merged_metadata: dict[str, object] = {
        "station_name": None,
        "station_code": None,
        "latitude": None,
        "longitude": None,
        "source_paths": {},
    }
    for year, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing source file for {year}: {path}")
        verbose_print(f"Loading dataset file for {year}: {path}", verbose=verbose, level=1)
        frame = parse_giro_2025(path)
        metadata = parse_dataset_header_metadata(path)
        frame["SourceYear"] = year
        frames.append(frame)
        merged_metadata["source_paths"][year] = str(path)
        for key in ("station_name", "station_code", "latitude", "longitude"):
            if merged_metadata.get(key) in (None, "") and metadata.get(key) not in (None, ""):
                merged_metadata[key] = metadata.get(key)

    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("Time")
        .drop_duplicates(subset=["Time"])
        .reset_index(drop=True)
    )
    df = df[(df["CS"] >= min_cs) | (df["CS"] == 999)].copy()

    for column in ["foF2", "foE", "foEs"]:
        if column in df.columns:
            df.loc[~df[column].between(*FREQ_BOUNDS), column] = np.nan
    if "MUFD" in df.columns:
        df.loc[~df["MUFD"].between(*MUFD_BOUNDS), "MUFD"] = np.nan
    for column in ["hmF2", "hF2", "hmE"]:
        if column in df.columns:
            df.loc[~df[column].between(*HEIGHT_BOUNDS), column] = np.nan
    df, mufd_restore_summary = restore_mufd_long_gaps(df, min_gap=MUFD_RESTORE_MIN_GAP)
    df = df.reset_index(drop=True)
    df.attrs["station_metadata"] = merged_metadata
    df.attrs["mufd_restore_records"] = mufd_restore_summary.to_dict("records")
    if not mufd_restore_summary.empty:
        verbose_print(
            "MUFD long-gap restoration "
            f"filled {int(mufd_restore_summary['RestoredRows'].sum())} row(s) "
            f"across {len(mufd_restore_summary)} gap(s).",
            verbose=verbose,
            level=1,
        )
    verbose_print(f"Loaded {len(df)} filtered rows from {len(paths)} dataset file(s).", verbose=verbose, level=1)
    return df


def _iter_true_runs(mask: pd.Series) -> Iterable[tuple[int, int]]:
    start_index: int | None = None
    for position, is_true in enumerate(mask.astype(bool).to_numpy()):
        if is_true and start_index is None:
            start_index = position
        elif not is_true and start_index is not None:
            yield start_index, position - 1
            start_index = None
    if start_index is not None:
        yield start_index, len(mask) - 1


def restore_mufd_long_gaps(
    df: pd.DataFrame,
    min_gap: pd.Timedelta = MUFD_RESTORE_MIN_GAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"Time", "foF2", "MUFD"}.issubset(df.columns):
        empty_summary = pd.DataFrame(
            columns=[
                "GapStart",
                "GapEnd",
                "GapDurationHours",
                "MissingRows",
                "FoF2Rows",
                "RestoredRows",
                "RatioRows",
                "AnalyticRows",
            ]
        )
        return df.copy(), empty_summary

    result = df.sort_values("Time").reset_index(drop=True).copy()
    step = infer_dataset_step(result)
    missing_mask = result["MUFD"].isna()
    if not missing_mask.any():
        empty_summary = pd.DataFrame(
            columns=[
                "GapStart",
                "GapEnd",
                "GapDurationHours",
                "MissingRows",
                "FoF2Rows",
                "RestoredRows",
                "RatioRows",
                "AnalyticRows",
            ]
        )
        return result, empty_summary

    time_index = pd.DatetimeIndex(result["Time"])
    ratio_series = (result["MUFD"] / result["foF2"]).where(result["foF2"] > 0)
    ratio_series = pd.Series(ratio_series.to_numpy(dtype=float), index=time_index)
    ratio_interp = ratio_series.interpolate(method="time", limit_area="inside")

    analytic_factor = None
    if "hmF2" in result.columns:
        analytic_values = np.where(result["hmF2"] + 176.0 > 0, 1490.0 / (result["hmF2"] + 176.0), np.nan)
        analytic_factor = pd.Series(analytic_values, index=time_index)

    records: list[dict[str, object]] = []
    for start_idx, end_idx in _iter_true_runs(missing_mask):
        gap_times = result.loc[start_idx : end_idx, "Time"]
        gap_duration = (gap_times.iloc[-1] - gap_times.iloc[0]) + step
        if gap_duration < min_gap:
            continue

        gap_frame = result.loc[start_idx:end_idx]
        restore_index = gap_frame.index[gap_frame["MUFD"].isna() & gap_frame["foF2"].notna()]
        if len(restore_index) == 0:
            continue

        restore_times = pd.DatetimeIndex(result.loc[restore_index, "Time"])
        ratio_factor = ratio_interp.reindex(restore_times)
        factor_used = ratio_factor.copy()
        ratio_rows = int(ratio_factor.notna().sum())
        analytic_rows = 0

        if analytic_factor is not None:
            missing_factor_mask = factor_used.isna()
            if missing_factor_mask.any():
                analytic_fill = analytic_factor.reindex(restore_times[missing_factor_mask.to_numpy()])
                factor_used.loc[missing_factor_mask] = analytic_fill.to_numpy()
                analytic_rows = int(np.count_nonzero(~pd.isna(analytic_fill.to_numpy())))

        valid_factor_mask = factor_used.notna() & (factor_used > 0)
        if not valid_factor_mask.any():
            continue

        valid_index = restore_index[valid_factor_mask.to_numpy()]
        restored_values = result.loc[valid_index, "foF2"].to_numpy(dtype=float) * factor_used.loc[valid_factor_mask].to_numpy(
            dtype=float
        )
        restored_values = np.where(
            (restored_values >= MUFD_BOUNDS[0]) & (restored_values <= MUFD_BOUNDS[1]),
            restored_values,
            np.nan,
        )
        valid_value_mask = ~np.isnan(restored_values)
        if not valid_value_mask.any():
            continue

        final_index = valid_index[valid_value_mask]
        result.loc[final_index, "MUFD"] = restored_values[valid_value_mask]
        records.append(
            {
                "GapStart": pd.Timestamp(gap_times.iloc[0]),
                "GapEnd": pd.Timestamp(gap_times.iloc[-1]),
                "GapDurationHours": round(float(gap_duration / pd.Timedelta(hours=1)), 3),
                "MissingRows": int(len(gap_frame)),
                "FoF2Rows": int(gap_frame["foF2"].notna().sum()),
                "RestoredRows": int(len(final_index)),
                "RatioRows": int(min(ratio_rows, len(final_index))),
                "AnalyticRows": int(max(0, len(final_index) - min(ratio_rows, len(final_index)))),
            }
        )

    return result, pd.DataFrame.from_records(
        records,
        columns=[
            "GapStart",
            "GapEnd",
            "GapDurationHours",
            "MissingRows",
            "FoF2Rows",
            "RestoredRows",
            "RatioRows",
            "AnalyticRows",
        ],
    )


def build_daily_frame(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    agg_columns = [column for column in ["foF2", "MUFD", "foE", "hmF2"] if column in df.columns]
    if not agg_columns:
        return pd.DataFrame(columns=["Time"])

    source = df.set_index("Time")[agg_columns].sort_index()
    if config.daily_approximate_input:
        dataset_step = resolve_dataset_step(df, config=config)
        source = (
            source.resample(dataset_step)
            .median()
            .interpolate(method="time", limit_direction="both")
        )

    return source.resample("1D").agg({column: "median" for column in agg_columns}).reset_index()


def _coerce_dataset_step(value: str | timedelta | pd.Timedelta) -> pd.Timedelta:
    step = pd.to_timedelta(value)
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError(f"dataset_step must be a positive duration, got {value!r}.")
    return step


def _snap_inferred_dataset_step(step: pd.Timedelta) -> pd.Timedelta:
    step = _coerce_dataset_step(step)
    snapped_value = int(round(step.value / DATASET_STEP_SNAP_GRID.value)) * DATASET_STEP_SNAP_GRID.value
    if snapped_value <= 0:
        return step
    snapped_step = pd.to_timedelta(snapped_value, unit="ns")
    if abs(step - snapped_step) <= DATASET_STEP_SNAP_TOLERANCE:
        return snapped_step
    return step


def _infer_dataset_step_from_quantized_diffs(positive_diffs: pd.Series) -> pd.Timedelta | None:
    if positive_diffs.empty:
        return None

    diff_seconds = positive_diffs.dt.total_seconds().to_numpy(dtype=float)
    if diff_seconds.size == 0:
        return None

    quantized_seconds = np.round(diff_seconds / 60.0).astype(int) * 60
    quantized_seconds = quantized_seconds[quantized_seconds >= int(MIN_REASONABLE_DATASET_STEP.total_seconds())]
    if quantized_seconds.size == 0:
        return None

    unique_seconds, counts = np.unique(quantized_seconds, return_counts=True)
    best_index = int(np.argmax(counts))
    return pd.to_timedelta(int(unique_seconds[best_index]), unit="s")


def infer_dataset_step(df: pd.DataFrame) -> pd.Timedelta:
    step_from_attrs = df.attrs.get("dataset_step")
    if step_from_attrs is not None:
        return _snap_inferred_dataset_step(_coerce_dataset_step(step_from_attrs))

    if "Time" not in df.columns:
        raise KeyError("The dataset must contain a 'Time' column to infer dataset_step.")

    time_values = (
        pd.Series(pd.to_datetime(df["Time"], utc=True, errors="coerce"))
        .dropna()
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(time_values) < 2:
        return DEFAULT_DATASET_STEP

    positive_diffs = time_values.diff().dropna()
    positive_diffs = positive_diffs[positive_diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return DEFAULT_DATASET_STEP

    diff_ns = positive_diffs.astype("int64").to_numpy(dtype=np.int64)
    inferred_step = pd.to_timedelta(int(np.median(diff_ns)), unit="ns")
    if inferred_step < MIN_REASONABLE_DATASET_STEP:
        quantized_step = _infer_dataset_step_from_quantized_diffs(positive_diffs)
        if quantized_step is not None:
            inferred_step = quantized_step
    return _snap_inferred_dataset_step(inferred_step)


def resolve_dataset_step(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.Timedelta:
    if config.dataset_step is not None:
        return _coerce_dataset_step(config.dataset_step)
    return infer_dataset_step(df)


def _duration_to_step_count(duration: pd.Timedelta, dataset_step: pd.Timedelta, label: str) -> int:
    if duration <= pd.Timedelta(0):
        raise ValueError(f"{label} must be positive, got {duration}.")
    if duration.value % dataset_step.value != 0:
        raise ValueError(
            f"{label}={duration} must be an exact multiple of dataset_step={dataset_step}."
        )
    return int(duration // dataset_step)


def resolve_target_label_source_column(
    target: str,
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> str | None:
    filtered_column = f"{target}_savgol"
    if config.use_filtered_target_labels and filtered_column in df.columns:
        return filtered_column
    if target in df.columns:
        return target
    if filtered_column in df.columns:
        return filtered_column
    return None


def iter_target_feature_bases(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[str, ...]:
    ordered_targets: list[str] = []
    for candidate in TARGET_LABELS:
        if candidate in df.columns or f"{candidate}_savgol" in df.columns:
            ordered_targets.append(candidate)
    for candidate in config.targets:
        if candidate not in ordered_targets and (candidate in df.columns or f"{candidate}_savgol" in df.columns):
            ordered_targets.append(candidate)
    return tuple(ordered_targets)


def build_dataset_frame(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    verbose: int | None = None,
) -> pd.DataFrame:
    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    dataset_step = resolve_dataset_step(df, config=config)
    forecast_steps = _duration_to_step_count(
        pd.Timedelta(hours=config.forecast_h),
        dataset_step,
        label="forecast_h",
    )
    df_h = (
        df.set_index("Time")
        .resample(dataset_step)
        .median(numeric_only=True)
        .reset_index()
        .sort_values("Time")
        .reset_index(drop=True)
    )
    df_h["hour"] = df_h["Time"].dt.hour
    df_h["month"] = df_h["Time"].dt.month
    df_h["doy"] = df_h["Time"].dt.dayofyear
    df_h["hour_sin"] = np.sin(2 * np.pi * df_h["hour"] / 24)
    df_h["hour_cos"] = np.cos(2 * np.pi * df_h["hour"] / 24)
    df_h["month_sin"] = np.sin(2 * np.pi * df_h["month"] / 12)
    df_h["month_cos"] = np.cos(2 * np.pi * df_h["month"] / 12)

    savgol_diagnostics = pd.DataFrame(columns=SAVGOL_DIAGNOSTIC_COLUMNS)
    if config.savgol_target_columns:
        df_h, savgol_diagnostics = apply_leak_safe_savgol_targets(
            df_h,
            target_columns=config.savgol_target_columns,
            time_col="Time",
            fit_end=resolve_savgol_fit_end(df_h, config=config),
            polyorder=config.savgol_polyorder,
        )

    target_series_sources: dict[str, str] = {}
    for target in iter_target_feature_bases(df_h, config=config):
        source_column = resolve_target_label_source_column(target, df_h, config=config)
        if source_column in df_h.columns:
            df_h[f"{target}_state"] = df_h[source_column]
            target_series_sources[target] = "filtered" if source_column.endswith("_savgol") else "raw"

    for target in config.targets:
        source_column = resolve_target_label_source_column(target, df_h, config=config)
        if source_column in df_h.columns:
            df_h[f"{target}_target"] = df_h[source_column].shift(-forecast_steps)
    df_h.attrs["dataset_step"] = dataset_step
    df_h.attrs["savgol_diagnostics_records"] = savgol_diagnostics.to_dict("records")
    df_h.attrs["target_series_sources"] = target_series_sources
    df_h.attrs["target_label_source"] = (
        "filtered" if config.use_filtered_target_labels and any(f"{target}_savgol" in df_h.columns for target in config.targets) else "raw"
    )
    savgol_metadata = build_savgol_metadata(savgol_diagnostics, config=config, df_h=df_h)
    df_h.attrs["savgol_metadata"] = savgol_metadata
    for message in _format_savgol_metadata_messages(savgol_metadata):
        verbose_print(message, verbose=effective_verbose, level=1)
    if target_series_sources:
        source_summary = ", ".join(f"{target}={source}" for target, source in target_series_sources.items())
        verbose_print(
            f"Target-derived ML feature sources: {source_summary}.",
            verbose=effective_verbose,
            level=1,
        )
    verbose_print(
        f"Target label source for ML/diagnostics: {df_h.attrs['target_label_source']}.",
        verbose=effective_verbose,
        level=1,
    )
    return df_h


def build_hourly_frame(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    verbose: int | None = None,
) -> pd.DataFrame:
    return build_dataset_frame(df, config=config, verbose=verbose)


def add_iri_baseline(
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    iri_config: IRIConfig = DEFAULT_IRI_CONFIG,
) -> pd.DataFrame:
    try:
        import PyIRI
        import PyIRI.main_library as ml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyIRI is required to compute the IRI baseline.") from exc

    alon = np.array([iri_config.lon_station])
    alat = np.array([iri_config.lat_station])
    aalt = np.array([iri_config.alt_km])

    def get_iri_day(year: int, month: int, day: int, f107_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hours = np.arange(0, 24, 1, dtype=float)
        f2, _, _, _, _, _, _ = ml.IRI_density_1day(
            year,
            month,
            day,
            hours,
            alon,
            alat,
            aalt,
            f107_val,
            PyIRI.coeff_dir,
        )
        fo_f2 = f2["fo"][:, 0]
        hm_f2 = f2["hm"][:, 0]
        m3000 = f2["M3000"][:, 0]
        mufd = m3000 * fo_f2
        return fo_f2, hm_f2, m3000, mufd

    iri_cache: dict[tuple[datetime.date, int], tuple[float, float, float, float]] = {}
    for current_date in pd.to_datetime(df_h["Time"]).dt.date.unique():
        try:
            f107_for_day = resolve_f107_adjusted_flux_for_date(current_date, fallback=iri_config.f107_val)
            fo_f2_day, hm_f2_day, m3000_day, mufd_day = get_iri_day(
                current_date.year,
                current_date.month,
                current_date.day,
                f107_for_day,
            )
            for hour in range(24):
                iri_cache[(current_date, hour)] = (
                    float(fo_f2_day[hour]) if hour < len(fo_f2_day) else np.nan,
                    float(hm_f2_day[hour]) if hour < len(hm_f2_day) else np.nan,
                    float(m3000_day[hour]) if hour < len(m3000_day) else np.nan,
                    float(mufd_day[hour]) if hour < len(mufd_day) else np.nan,
                )
        except Exception:
            for hour in range(24):
                iri_cache[(current_date, hour)] = (np.nan, np.nan, np.nan, np.nan)

    def lookup(timestamp: pd.Timestamp) -> tuple[float, float, float, float]:
        date_key = pd.Timestamp(timestamp).date()
        hour_key = pd.Timestamp(timestamp).hour
        return iri_cache.get((date_key, hour_key), (np.nan, np.nan, np.nan, np.nan))

    dataset_step = resolve_dataset_step(df_h, config=config)
    forecast_steps = _duration_to_step_count(
        pd.Timedelta(hours=config.forecast_h),
        dataset_step,
        label="forecast_h",
    )
    result = df_h.copy()
    result["IRI_foF2"] = result["Time"].map(lambda value: lookup(value)[0])
    result["IRI_hmF2"] = result["Time"].map(lambda value: lookup(value)[1])
    result["IRI_M3000"] = result["Time"].map(lambda value: lookup(value)[2])
    result["IRI_MUFD"] = result["Time"].map(lambda value: lookup(value)[3])
    result["IRI_foF2_pred"] = result["IRI_foF2"].shift(-forecast_steps)
    result["IRI_M3000_pred"] = result["IRI_M3000"].shift(-forecast_steps)
    result["IRI_MUFD_pred"] = result["IRI_MUFD"].shift(-forecast_steps)
    result.attrs["dataset_step"] = dataset_step
    return result


def resolve_iri_config_from_metadata(
    station_metadata: dict[str, object] | None,
    iri_config: IRIConfig = DEFAULT_IRI_CONFIG,
    verbose: int = 0,
) -> IRIConfig:
    resolved_fields: dict[str, float] = {}
    if station_metadata:
        latitude = station_metadata.get("latitude")
        longitude = station_metadata.get("longitude")
        if latitude is not None and longitude is not None:
            resolved_fields["lat_station"] = float(latitude)
            resolved_fields["lon_station"] = float(longitude)
            try:
                resolved_fields["alt_km"] = fetch_station_altitude_km(
                    resolved_fields["lat_station"],
                    resolved_fields["lon_station"],
                )
                verbose_print(
                    f"Fetched station altitude: {resolved_fields['alt_km']:.3f} km",
                    verbose=verbose,
                    level=1,
                )
            except Exception as exc:
                verbose_print(
                    f"Open-Elevation lookup failed ({exc}); keeping alt_km={iri_config.alt_km}.",
                    verbose=verbose,
                    level=1,
                )

    if not resolved_fields:
        return iri_config
    return replace(iri_config, **resolved_fields)


def add_analytic_baseline(df_h: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    dataset_step = resolve_dataset_step(df_h, config=config)
    forecast_steps = _duration_to_step_count(
        pd.Timedelta(hours=config.forecast_h),
        dataset_step,
        label="forecast_h",
    )
    result = df_h.copy()
    nm_f2 = np.maximum(result["TEC"] * 1e16 / (config.tau_km * 1e3), 0)
    result["anal_foF2"] = np.sqrt(nm_f2 / 1.24e10)
    result["anal_M3000"] = np.where(result["hmF2"] + 176 > 0, 1490.0 / (result["hmF2"] + 176.0), np.nan)
    result["anal_MUFD"] = result["anal_M3000"] * result["anal_foF2"]
    result["anal_foF2_pred"] = result["anal_foF2"].shift(-forecast_steps)
    result["anal_MUFD_pred"] = result["anal_MUFD"].shift(-forecast_steps)
    result.attrs["dataset_step"] = dataset_step
    return result


def build_yearly_season_bands(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    years = sorted(pd.Series(df["Time"]).dt.year.dropna().astype(int).unique())
    season_bands: list[tuple[str, str, str, str]] = []
    for year in years:
        season_bands.extend(
            [
                (f"Зима {year - 1}/{str(year)[-2:]}", f"{year - 1}-12-01", f"{year}-02-28", "rgba(100,149,237,0.15)"),
                (f"Весна {year}", f"{year}-03-01", f"{year}-05-31", "rgba(60,179,113,0.15)"),
                (f"Лето {year}", f"{year}-06-01", f"{year}-08-31", "rgba(218,165,32,0.20)"),
                (f"Осень {year}", f"{year}-09-01", f"{year}-11-30", "rgba(210,105,30,0.15)"),
            ]
        )
    return season_bands


def resolve_verbose(config: PipelineConfig = DEFAULT_CONFIG, verbose: int | None = None) -> int:
    return config.verbose if verbose is None else verbose


def verbose_print(message: str, verbose: int, level: int = 1) -> None:
    if verbose >= level:
        print(message)


@dataclass
class ProgressTracker:
    total: int
    verbose: int
    label: str
    completed: int = 0
    last_percent: int = -1

    def advance(self, count: int = 1, detail: str | None = None) -> None:
        if self.total < 1 or count < 1:
            return

        self.completed = min(self.total, self.completed + count)
        percent = int(round(self.completed * 100 / self.total))
        if percent == self.last_percent and self.completed < self.total:
            return

        filled = min(24, int(self.completed * 24 / self.total))
        bar = "#" * filled + "-" * (24 - filled)
        remaining = max(self.total - self.completed, 0)
        detail_suffix = f" | {detail}" if detail else ""
        verbose_print(
            f"{self.label} [{bar}] {percent:3d}% ({self.completed}/{self.total}, remaining={remaining}){detail_suffix}",
            verbose=self.verbose,
            level=1,
        )
        self.last_percent = percent


def _format_savgol_diagnostics_messages(savgol_diagnostics: pd.DataFrame) -> list[str]:
    if savgol_diagnostics.empty:
        return []

    messages: list[str] = []
    for record in savgol_diagnostics.to_dict("records"):
        variance_ratio = record.get("VarianceRatio")
        variance_text = "nan" if pd.isna(variance_ratio) else f"{float(variance_ratio):.3f}"
        messages.append(
            "Savitzky-Golay smoothing "
            f"[{record.get('Column')} -> {record.get('OutputColumn')}]: "
            f"applied={bool(record.get('Applied'))}, "
            f"polyorder={record.get('Polyorder')}, "
            f"window={record.get('WindowLength')}, "
            f"corr_steps={record.get('CorrelationLengthSteps')}, "
            f"fit_rows={record.get('FitRows')}, "
            f"variance_ratio={variance_text}, "
            f"reason={record.get('Reason')}."
        )
    return messages


def _normalize_metadata_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.tz_localize("UTC").isoformat()
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_savgol_metadata(
    savgol_diagnostics: pd.DataFrame,
    *,
    config: PipelineConfig,
    df_h: pd.DataFrame,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for record in savgol_diagnostics.to_dict("records"):
        records.append(
            {
                "column": str(record.get("Column")),
                "output_column": str(record.get("OutputColumn")),
                "applied": bool(record.get("Applied")),
                "reason": _normalize_metadata_scalar(record.get("Reason")),
                "fit_rows": _normalize_metadata_scalar(record.get("FitRows")),
                "fit_start": _normalize_metadata_scalar(record.get("FitStart")),
                "fit_end": _normalize_metadata_scalar(record.get("FitEnd")),
                "correlation_length_steps": _normalize_metadata_scalar(record.get("CorrelationLengthSteps")),
                "window_length": _normalize_metadata_scalar(record.get("WindowLength")),
                "polyorder": _normalize_metadata_scalar(record.get("Polyorder")),
                "variance_ratio": _normalize_metadata_scalar(record.get("VarianceRatio")),
                "peak_count_raw": _normalize_metadata_scalar(record.get("PeakCountRaw")),
                "peak_count_smooth": _normalize_metadata_scalar(record.get("PeakCountSmooth")),
                "median_peak_shift_steps": _normalize_metadata_scalar(record.get("MedianPeakShiftSteps")),
                "derivative_std_raw": _normalize_metadata_scalar(record.get("DerivativeStdRaw")),
                "derivative_std_smooth": _normalize_metadata_scalar(record.get("DerivativeStdSmooth")),
                "turning_points_raw": _normalize_metadata_scalar(record.get("TurningPointsRaw")),
                "turning_points_smooth": _normalize_metadata_scalar(record.get("TurningPointsSmooth")),
                "suggestion": _normalize_metadata_scalar(record.get("Suggestion")),
            }
        )

    return {
        "enabled": bool(config.savgol_target_columns),
        "requested_target_columns": [str(value) for value in config.savgol_target_columns],
        "polyorder": int(config.savgol_polyorder),
        "use_filtered_target_labels": bool(config.use_filtered_target_labels),
        "target_label_source": str(df_h.attrs.get("target_label_source", "raw")),
        "target_series_sources": {str(key): str(value) for key, value in dict(df_h.attrs.get("target_series_sources", {})).items()},
        "columns": records,
    }


def _format_savgol_metadata_messages(savgol_metadata: dict[str, object]) -> list[str]:
    requested_targets = tuple(savgol_metadata.get("requested_target_columns", []))
    messages = [
        "Savitzky-Golay final settings: "
        f"enabled={bool(savgol_metadata.get('enabled'))}, "
        f"requested_targets={requested_targets}, "
        f"polyorder={savgol_metadata.get('polyorder')}, "
        f"use_filtered_target_labels={bool(savgol_metadata.get('use_filtered_target_labels'))}, "
        f"target_label_source={savgol_metadata.get('target_label_source')}."
    ]
    for record in savgol_metadata.get("columns", []):
        messages.append(
            "Savitzky-Golay final parameters "
            f"[{record.get('column')} -> {record.get('output_column')}]: "
            f"applied={bool(record.get('applied'))}, "
            f"fit_start={record.get('fit_start')}, "
            f"fit_end={record.get('fit_end')}, "
            f"fit_rows={record.get('fit_rows')}, "
            f"window={record.get('window_length')}, "
            f"polyorder={record.get('polyorder')}, "
            f"corr_steps={record.get('correlation_length_steps')}, "
            f"variance_ratio={record.get('variance_ratio')}, "
            f"peak_raw={record.get('peak_count_raw')}, "
            f"peak_smooth={record.get('peak_count_smooth')}, "
            f"turning_raw={record.get('turning_points_raw')}, "
            f"turning_smooth={record.get('turning_points_smooth')}, "
            f"deriv_std_raw={record.get('derivative_std_raw')}, "
            f"deriv_std_smooth={record.get('derivative_std_smooth')}, "
            f"median_peak_shift={record.get('median_peak_shift_steps')}, "
            f"reason={record.get('reason')}, "
            f"suggestion={record.get('suggestion')}."
        )
    return messages


def is_future_derived_column(column: str) -> bool:
    return column.endswith("_target") or column.endswith("_pred") or column.endswith("_savgol")


def get_feature_pool(df_h: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df_h.select_dtypes(include=np.number).columns
        if column not in NON_FEATURE_COLUMNS and not is_future_derived_column(column)
    ]


def select_features(
    train_frame: pd.DataFrame,
    feature_pool: Iterable[str],
    min_coverage: float,
) -> list[str]:
    selected = []
    for feature in feature_pool:
        if feature not in train_frame.columns:
            continue
        if train_frame[feature].notna().mean() >= min_coverage:
            selected.append(feature)
    return selected


def make_models(config: PipelineConfig = DEFAULT_CONFIG) -> dict[str, object]:
    models: dict[str, object] = {
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            n_jobs=config.model_n_jobs,
            random_state=42,
        ),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            n_jobs=config.model_n_jobs,
            verbosity=0,
            random_state=42,
        )
    return models


def _load_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("optuna is required to run the Optuna walk-forward workflow.") from exc
    return optuna


def _configure_optuna_logging(optuna_module, verbose: int) -> None:
    logging_api = getattr(optuna_module, "logging", None)
    if logging_api is None or not hasattr(logging_api, "set_verbosity"):
        return

    target_level = getattr(logging_api, "INFO", None) if verbose >= 2 else getattr(logging_api, "WARNING", None)
    if target_level is not None:
        logging_api.set_verbosity(target_level)


def _resolve_optuna_metric(metric_name: str) -> tuple[str, str, Callable[[pd.Series, np.ndarray], float]]:
    normalized = metric_name.strip().upper()
    metric_map: dict[str, tuple[str, str, Callable[[pd.Series, np.ndarray], float]]] = {
        "MAE": ("MAE", "minimize", lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred))),
        "RMSE": (
            "RMSE",
            "minimize",
            lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        ),
        "MAPE": (
            "MAPE_%",
            "minimize",
            lambda y_true, y_pred: float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        ),
        "MAPE_%": (
            "MAPE_%",
            "minimize",
            lambda y_true, y_pred: float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        ),
        "R2": ("R2", "maximize", lambda y_true, y_pred: float(r2_score(y_true, y_pred))),
    }
    if normalized not in metric_map:
        raise ValueError(f"Unsupported Optuna metric {metric_name!r}. Use one of: {', '.join(sorted(metric_map))}.")
    return metric_map[normalized]


def _resolve_optuna_models(model_names: Iterable[str]) -> list[str]:
    supported = {"ElasticNet", "RandomForest", "XGBoost"}
    resolved = []
    for model_name in model_names:
        if model_name not in supported:
            raise ValueError(
                f"Unsupported Optuna model {model_name!r}. Use one of: {', '.join(sorted(supported))}."
            )
        if model_name == "XGBoost" and xgb is None:
            continue
        if model_name not in resolved:
            resolved.append(model_name)
    if not resolved:
        raise ValueError("No Optuna models are available. Install xgboost or choose supported sklearn models.")
    return resolved


def _sample_optuna_params(model_name: str, trial) -> dict[str, Any]:
    if model_name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("XGBoost is not installed.")
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0),
        }
    raise ValueError(f"Unsupported Optuna model {model_name!r}.")


def _build_optuna_estimator(model_name: str, params: dict[str, Any], random_state: int, n_jobs: int) -> object:
    if model_name == "ElasticNet":
        model_params = {
            "alpha": float(params["alpha"]),
            "l1_ratio": float(params["l1_ratio"]),
            "max_iter": 10000,
            "random_state": random_state,
        }
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**model_params)),
            ]
        )
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params["min_samples_split"]),
            max_features=float(params["max_features"]),
            bootstrap=bool(params["bootstrap"]),
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("XGBoost is not installed.")
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=int(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            gamma=float(params["gamma"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=n_jobs,
            verbosity=0,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported Optuna model {model_name!r}.")


def iter_daily_windows(seasons: dict[str, tuple[datetime, datetime]]) -> Iterable[tuple[str, datetime]]:
    for season_name, (season_start, season_end) in seasons.items():
        current_day = season_start
        while current_day <= season_end:
            yield season_name, current_day
            current_day += timedelta(days=1)


def _coerce_date_boundary(value: date | datetime | str | pd.Timestamp, label: str) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid date, got {value!r}.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.date()


def resolve_ml_date_range(config: PipelineConfig = DEFAULT_CONFIG) -> tuple[date | None, date | None]:
    date_start = None if config.ml_date_start is None else _coerce_date_boundary(config.ml_date_start, "ml_date_start")
    date_end = None if config.ml_date_end is None else _coerce_date_boundary(config.ml_date_end, "ml_date_end")
    if date_start is not None and date_end is not None and date_start > date_end:
        raise ValueError(f"ml_date_start={date_start} must be on or before ml_date_end={date_end}.")
    return date_start, date_end


def iter_configured_daily_windows(config: PipelineConfig = DEFAULT_CONFIG) -> Iterable[tuple[str, datetime]]:
    date_start, date_end = resolve_ml_date_range(config=config)
    for season_name, current_day in iter_daily_windows(config.seasons):
        current_date = pd.Timestamp(current_day).date()
        if date_start is not None and current_date < date_start:
            continue
        if date_end is not None and current_date > date_end:
            continue
        yield season_name, current_day


def count_planned_model_iterations(
    config: PipelineConfig,
    model_names: Iterable[str],
) -> int:
    day_count = sum(1 for _ in iter_configured_daily_windows(config))
    return len(tuple(config.window_list)) * day_count * len(tuple(config.targets)) * len(tuple(model_names))


def build_window_file_prefix(train_days: int) -> str:
    return f"window_{int(train_days)}d"


def resolve_window_parallel_settings(
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[int, int]:
    windows = tuple(config.window_list)
    window_count = max(1, len(windows))
    cpu_total = max(1, os.cpu_count() or 1)
    window_workers = min(window_count, cpu_total)
    model_n_jobs = max(1, cpu_total // window_count)
    return window_workers, model_n_jobs


def _as_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def resolve_savgol_fit_end(df_h: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.Timestamp | None:
    if df_h.empty or "Time" not in df_h.columns:
        return None

    dataset_step = resolve_dataset_step(df_h, config=config)
    candidate_days = sorted(_as_utc_timestamp(current_day) for _, current_day in iter_configured_daily_windows(config))
    if not candidate_days:
        return None

    valid_times = pd.to_datetime(df_h["Time"], utc=True, errors="coerce").dropna()
    if valid_times.empty:
        return None
    data_start = _as_utc_timestamp(valid_times.min())
    earliest_eligible_day = data_start.floor("D") + pd.Timedelta(days=min(config.window_list or (0,)))

    for candidate_day in candidate_days:
        if candidate_day >= earliest_eligible_day:
            return candidate_day - dataset_step
    return _as_utc_timestamp(valid_times.max())


def split_walk_forward_window(
    df_h: pd.DataFrame,
    current_day: datetime,
    train_days: int,
    forecast_h: int,
    dataset_step: str | timedelta | pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    resolved_dataset_step = _coerce_dataset_step(dataset_step) if dataset_step is not None else infer_dataset_step(df_h)
    forecast_delta = pd.Timedelta(hours=forecast_h)
    _duration_to_step_count(forecast_delta, resolved_dataset_step, label="forecast_h")
    test_start = _as_utc_timestamp(current_day)
    test_end = test_start + pd.Timedelta(days=1) - resolved_dataset_step
    train_end = test_start - resolved_dataset_step
    train_start = train_end - pd.Timedelta(days=train_days) + resolved_dataset_step
    safe_train_end = train_end - forecast_delta

    train_frame = df_h[(df_h["Time"] >= train_start) & (df_h["Time"] <= safe_train_end)].copy()
    test_frame = df_h[(df_h["Time"] >= test_start) & (df_h["Time"] <= test_end)].copy()
    return train_frame, test_frame, {
        "train_start": train_start,
        "train_end": train_end,
        "safe_train_end": safe_train_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def split_train_val_test_window(
    df_h: pd.DataFrame,
    current_day: datetime,
    train_days: int,
    forecast_h: int,
    val_days: int = 1,
    test_h: int = 24,
    dataset_step: str | timedelta | pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    if train_days < 1:
        raise ValueError("train_days must be at least 1.")
    if val_days < 1:
        raise ValueError("val_days must be at least 1.")
    if test_h < 1:
        raise ValueError("test_h must be at least 1.")

    resolved_dataset_step = _coerce_dataset_step(dataset_step) if dataset_step is not None else infer_dataset_step(df_h)
    forecast_delta = pd.Timedelta(hours=forecast_h)
    _duration_to_step_count(forecast_delta, resolved_dataset_step, label="forecast_h")
    test_delta = pd.Timedelta(hours=test_h)
    if test_delta < resolved_dataset_step:
        raise ValueError(f"test_h must be at least one dataset_step ({resolved_dataset_step}).")

    test_start = _as_utc_timestamp(current_day)
    test_end = test_start + test_delta - resolved_dataset_step
    val_end = test_start - forecast_delta - resolved_dataset_step
    val_start = val_end - pd.Timedelta(days=val_days) + resolved_dataset_step
    train_end = val_start - resolved_dataset_step
    train_start = train_end - pd.Timedelta(days=train_days) + resolved_dataset_step
    safe_train_end = train_end - forecast_delta

    train_frame = df_h[(df_h["Time"] >= train_start) & (df_h["Time"] <= safe_train_end)].copy()
    val_frame = df_h[(df_h["Time"] >= val_start) & (df_h["Time"] <= val_end)].copy()
    test_frame = df_h[(df_h["Time"] >= test_start) & (df_h["Time"] <= test_end)].copy()
    return train_frame, val_frame, test_frame, {
        "train_start": train_start,
        "train_end": train_end,
        "safe_train_end": safe_train_end,
        "val_start": val_start,
        "val_end": val_end,
        "val_label_start": val_start + forecast_delta,
        "val_label_end": val_end + forecast_delta,
        "test_start": test_start,
        "test_end": test_end,
    }


def run_walk_forward_models(
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    verbose: int | None = None,
    iteration_exporter: Callable[[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]], None]
    | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    feature_pool = get_feature_pool(df_h)
    forecast_delta = pd.Timedelta(hours=config.forecast_h)
    model_names = tuple(make_models(config=config).keys())
    progress = ProgressTracker(
        total=count_planned_model_iterations(config=config, model_names=model_names),
        verbose=effective_verbose,
        label="Walk-forward progress",
    )
    all_metrics: list[dict[str, object]] = []
    all_importance: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []

    verbose_print(
        f"Walk-forward modeling started: windows={config.window_list}, targets={config.targets}.",
        verbose=effective_verbose,
        level=1,
    )
    for train_days in config.window_list:
        metrics_before_window = len(all_metrics)
        for season_name, current_day in iter_configured_daily_windows(config):
            train_frame, test_frame, _ = split_walk_forward_window(
                df_h=df_h,
                current_day=current_day,
                train_days=train_days,
                forecast_h=config.forecast_h,
            )

            if len(train_frame) < config.min_train_rows or test_frame.empty:
                verbose_print(
                    f"Skipping {current_day:%Y-%m-%d} for {train_days}d window: insufficient train/test rows.",
                    verbose=effective_verbose,
                    level=2,
                )
                progress.advance(
                    len(config.targets) * len(model_names),
                    detail=f"{current_day:%Y-%m-%d} train={train_days}d skipped: train/test rows",
                )
                continue

            split_features = select_features(train_frame, feature_pool, config.min_feature_coverage)
            if not split_features:
                verbose_print(
                    f"Skipping {current_day:%Y-%m-%d} for {train_days}d window: no features passed coverage filter.",
                    verbose=effective_verbose,
                    level=2,
                )
                progress.advance(
                    len(config.targets) * len(model_names),
                    detail=f"{current_day:%Y-%m-%d} train={train_days}d skipped: no features",
                )
                continue

            imputer = SimpleImputer(strategy="median")
            train_imputed = pd.DataFrame(
                imputer.fit_transform(train_frame[split_features]),
                columns=split_features,
                index=train_frame.index,
            )
            test_imputed = pd.DataFrame(
                imputer.transform(test_frame[split_features]),
                columns=split_features,
                index=test_frame.index,
            )

            for target in config.targets:
                target_column = f"{target}_target"
                if target_column not in df_h.columns:
                    progress.advance(
                        len(model_names),
                        detail=f"{current_day:%Y-%m-%d} train={train_days}d target={target} skipped: missing labels",
                    )
                    continue

                y_train = train_frame[target_column]
                y_test = test_frame[target_column]
                valid_train_mask = y_train.notna()
                valid_test_mask = y_test.notna()

                if valid_train_mask.sum() < config.min_train_target_rows or valid_test_mask.sum() < config.min_eval_rows:
                    verbose_print(
                        f"Skipping {current_day:%Y-%m-%d} target={target} for {train_days}d window: "
                        "insufficient labeled train/test rows.",
                        verbose=effective_verbose,
                        level=2,
                    )
                    progress.advance(
                        len(model_names),
                        detail=f"{current_day:%Y-%m-%d} train={train_days}d target={target} skipped: label rows",
                    )
                    continue

                x_train_fit = train_imputed.loc[valid_train_mask]
                y_train_fit = y_train.loc[valid_train_mask]
                x_test_eval = test_imputed.loc[valid_test_mask]
                y_test_eval = y_test.loc[valid_test_mask]

                for model_name, model in make_models(config=config).items():
                    model.fit(x_train_fit, y_train_fit)
                    y_pred = model.predict(x_test_eval)

                    mae = mean_absolute_error(y_test_eval, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
                    r2 = r2_score(y_test_eval, y_pred)
                    mape = mean_absolute_percentage_error(y_test_eval, y_pred) * 100
                    cc = np.corrcoef(y_test_eval, y_pred)[0, 1] if len(y_test_eval) > 1 else np.nan

                    all_metrics.append(
                        {
                            "TrainDays": train_days,
                            "Season": season_name,
                            "Date": current_day.strftime("%Y-%m-%d"),
                            "Target": target,
                            "Model": model_name,
                            "MAE": round(float(mae), 4),
                            "RMSE": round(float(rmse), 4),
                            "R2": round(float(r2), 4),
                            "MAPE_%": round(float(mape), 2),
                            "CC": round(float(cc), 4) if not np.isnan(cc) else np.nan,
                            "N_train": int(len(x_train_fit)),
                            "FeatureCount": int(len(split_features)),
                        }
                    )

                    for time_value, actual_value, predicted_value in zip(
                        test_frame.loc[valid_test_mask, "Time"],
                        y_test_eval,
                        y_pred,
                    ):
                        source_time = pd.Timestamp(time_value)
                        all_predictions.append(
                            {
                                "TrainDays": train_days,
                                "Date": current_day.strftime("%Y-%m-%d"),
                                "Season": season_name,
                                "Time": source_time,
                                "TargetTime": source_time + forecast_delta,
                                "Target": target,
                                "Model": model_name,
                                "actual": float(actual_value),
                                "predicted": float(predicted_value),
                            }
                        )

                    if hasattr(model, "feature_importances_"):
                        importance_values = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        importance_values = np.abs(model.coef_)
                    else:
                        importance_values = np.zeros(len(split_features))

                    for feature, importance in zip(split_features, importance_values):
                        all_importance.append(
                            {
                                "TrainDays": train_days,
                                "Season": season_name,
                                "Date": current_day.strftime("%Y-%m-%d"),
                                "Target": target,
                                "Model": model_name,
                                "Feature": feature,
                                "Importance": round(float(importance), 6),
                            }
                        )
                    if iteration_exporter is not None:
                        iteration_exporter(all_metrics, all_importance, all_predictions)
                    progress.advance(
                        detail=(
                            f"{current_day:%Y-%m-%d} train={train_days}d "
                            f"target={target} model={model_name}"
                        ),
                    )
        verbose_print(
            f"Completed {train_days}d walk-forward window: added {len(all_metrics) - metrics_before_window} metric rows.",
            verbose=effective_verbose,
            level=1,
        )

    metrics_df = pd.DataFrame(all_metrics)
    fi_df = pd.DataFrame(all_importance)
    preds_df = pd.DataFrame(all_predictions)
    if not metrics_df.empty:
        metrics_df["Date"] = pd.to_datetime(metrics_df["Date"])
    if not fi_df.empty:
        fi_df["Date"] = pd.to_datetime(fi_df["Date"])
    verbose_print(
        f"Walk-forward modeling finished: metrics={len(metrics_df)}, importance={len(fi_df)}, predictions={len(preds_df)}.",
        verbose=effective_verbose,
        level=1,
    )
    return metrics_df, fi_df, preds_df


def run_optuna_walk_forward_models(
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG,
    verbose: int | None = None,
    iteration_exporter: Callable[[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]], None]
    | None = None,
    shap_record_sink: list[dict[str, object]] | None = None,
    shap_iteration_exporter: Callable[[pd.DataFrame, dict[str, object]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    optuna = _load_optuna()
    if optuna_config.n_trials < 1:
        raise ValueError("Optuna n_trials must be at least 1.")

    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    _configure_optuna_logging(optuna, effective_verbose)
    feature_pool = get_feature_pool(df_h)
    forecast_delta = pd.Timedelta(hours=config.forecast_h)
    resolved_val_days = optuna_config.val_days or max(1, (optuna_config.test_h + 23) // 24)
    resolved_models = _resolve_optuna_models(optuna_config.models)
    progress = ProgressTracker(
        total=count_planned_model_iterations(config=config, model_names=resolved_models),
        verbose=effective_verbose,
        label="Optuna progress",
    )
    metric_name, study_direction, objective_scorer = _resolve_optuna_metric(optuna_config.metric)

    all_metrics: list[dict[str, object]] = []
    all_trials: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []

    verbose_print(
        "Optuna walk-forward modeling started: "
        f"windows={config.window_list}, targets={config.targets}, val_days={resolved_val_days}, "
        f"test_h={optuna_config.test_h}, trials={optuna_config.n_trials}, "
        f"metric={metric_name}, models={resolved_models}.",
        verbose=effective_verbose,
        level=1,
    )
    for train_days in config.window_list:
        metrics_before_window = len(all_metrics)
        for season_name, current_day in iter_configured_daily_windows(config):
            train_frame, val_frame, test_frame, _ = split_train_val_test_window(
                df_h=df_h,
                current_day=current_day,
                train_days=train_days,
                forecast_h=config.forecast_h,
                val_days=resolved_val_days,
                test_h=optuna_config.test_h,
            )

            if len(train_frame) < config.min_train_rows or val_frame.empty or test_frame.empty:
                verbose_print(
                    f"Skipping {current_day:%Y-%m-%d} for {train_days}d window: insufficient train/val/test rows.",
                    verbose=effective_verbose,
                    level=2,
                )
                progress.advance(
                    len(config.targets) * len(resolved_models),
                    detail=f"{current_day:%Y-%m-%d} train={train_days}d skipped: train/val/test rows",
                )
                continue

            split_features = select_features(train_frame, feature_pool, config.min_feature_coverage)
            if not split_features:
                verbose_print(
                    f"Skipping {current_day:%Y-%m-%d} for {train_days}d window: no features passed coverage filter.",
                    verbose=effective_verbose,
                    level=2,
                )
                progress.advance(
                    len(config.targets) * len(resolved_models),
                    detail=f"{current_day:%Y-%m-%d} train={train_days}d skipped: no features",
                )
                continue

            train_imputer = SimpleImputer(strategy="median")
            train_imputed = pd.DataFrame(
                train_imputer.fit_transform(train_frame[split_features]),
                columns=split_features,
                index=train_frame.index,
            )
            val_imputed = pd.DataFrame(
                train_imputer.transform(val_frame[split_features]),
                columns=split_features,
                index=val_frame.index,
            )
            test_imputed = pd.DataFrame(
                train_imputer.transform(test_frame[split_features]),
                columns=split_features,
                index=test_frame.index,
            )

            for target in config.targets:
                target_column = f"{target}_target"
                if target_column not in df_h.columns:
                    progress.advance(
                        len(resolved_models),
                        detail=f"{current_day:%Y-%m-%d} train={train_days}d target={target} skipped: missing labels",
                    )
                    continue

                y_train = train_frame[target_column]
                y_val = val_frame[target_column]
                y_test = test_frame[target_column]

                valid_train_mask = y_train.notna()
                valid_val_mask = y_val.notna()
                valid_test_mask = y_test.notna()

                if (
                    valid_train_mask.sum() < config.min_train_target_rows
                    or valid_val_mask.sum() < config.min_eval_rows
                    or valid_test_mask.sum() < config.min_eval_rows
                ):
                    verbose_print(
                        f"Skipping {current_day:%Y-%m-%d} target={target} for {train_days}d window: "
                        "insufficient labeled train/val/test rows.",
                        verbose=effective_verbose,
                        level=2,
                    )
                    progress.advance(
                        len(resolved_models),
                        detail=f"{current_day:%Y-%m-%d} train={train_days}d target={target} skipped: label rows",
                    )
                    continue

                x_train_fit = train_imputed.loc[valid_train_mask]
                y_train_fit = y_train.loc[valid_train_mask]
                x_val_eval = val_imputed.loc[valid_val_mask]
                y_val_eval = y_val.loc[valid_val_mask]
                x_test_eval = test_imputed.loc[valid_test_mask]
                y_test_eval = y_test.loc[valid_test_mask]

                for model_name in resolved_models:
                    sampler = optuna.samplers.TPESampler(seed=optuna_config.random_state)
                    study = optuna.create_study(direction=study_direction, sampler=sampler)

                    def objective(trial) -> float:
                        params = _sample_optuna_params(model_name, trial)
                        model = _build_optuna_estimator(
                            model_name,
                            params,
                            optuna_config.random_state,
                            config.model_n_jobs,
                        )
                        model.fit(x_train_fit, y_train_fit)
                        y_val_pred = model.predict(x_val_eval)
                        score = objective_scorer(y_val_eval, y_val_pred)
                        if np.isfinite(score):
                            return score
                        return np.inf if study_direction == "minimize" else -np.inf

                    study.optimize(objective, n_trials=optuna_config.n_trials)

                    best_params = dict(study.best_params)
                    for trial in getattr(study, "trials", []):
                        trial_value = getattr(trial, "value", np.nan)
                        all_trials.append(
                            {
                                "TrainDays": train_days,
                                "ValDays": resolved_val_days,
                                "TestH": optuna_config.test_h,
                                "Season": season_name,
                                "Date": current_day.strftime("%Y-%m-%d"),
                                "Target": target,
                                "Model": model_name,
                                "TrialNumber": getattr(trial, "number", np.nan),
                                "Value": round(float(trial_value), 6) if trial_value is not None else np.nan,
                                "Params": json.dumps(dict(getattr(trial, "params", {})), sort_keys=True),
                                "State": str(getattr(trial, "state", "")),
                            }
                        )

                    final_train_frame = train_frame
                    if optuna_config.train_final_on_train_val:
                        final_train_frame = pd.concat([train_frame, val_frame], axis=0)
                    final_y = final_train_frame[target_column]
                    valid_final_mask = final_y.notna()

                    final_imputer = SimpleImputer(strategy="median")
                    final_imputed = pd.DataFrame(
                        final_imputer.fit_transform(final_train_frame[split_features]),
                        columns=split_features,
                        index=final_train_frame.index,
                    )
                    final_test_imputed = pd.DataFrame(
                        final_imputer.transform(test_frame[split_features]),
                        columns=split_features,
                        index=test_frame.index,
                    )

                    x_final_fit = final_imputed.loc[valid_final_mask]
                    y_final_fit = final_y.loc[valid_final_mask]
                    x_test_final = final_test_imputed.loc[valid_test_mask]

                    final_model = _build_optuna_estimator(
                        model_name,
                        best_params,
                        optuna_config.random_state,
                        config.model_n_jobs,
                    )
                    final_model.fit(x_final_fit, y_final_fit)
                    y_pred = final_model.predict(x_test_final)

                    mae = mean_absolute_error(y_test_eval, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
                    r2 = r2_score(y_test_eval, y_pred)
                    mape = mean_absolute_percentage_error(y_test_eval, y_pred) * 100
                    cc = np.corrcoef(y_test_eval, y_pred)[0, 1] if len(y_test_eval) > 1 else np.nan

                    all_metrics.append(
                        {
                            "TrainDays": train_days,
                            "ValDays": resolved_val_days,
                            "TestH": optuna_config.test_h,
                            "Season": season_name,
                            "Date": current_day.strftime("%Y-%m-%d"),
                            "Target": target,
                            "Model": model_name,
                            "OptunaMetric": metric_name,
                            "BestValScore": round(float(study.best_value), 6),
                            "BestParams": json.dumps(best_params, sort_keys=True),
                            "MAE": round(float(mae), 4),
                            "RMSE": round(float(rmse), 4),
                            "R2": round(float(r2), 4),
                            "MAPE_%": round(float(mape), 2),
                            "CC": round(float(cc), 4) if not np.isnan(cc) else np.nan,
                            "N_train": int(valid_train_mask.sum()),
                            "N_val": int(valid_val_mask.sum()),
                            "N_test": int(valid_test_mask.sum()),
                            "N_final_train": int(valid_final_mask.sum()),
                            "FeatureCount": int(len(split_features)),
                            "Trials": int(optuna_config.n_trials),
                        }
                    )

                    for time_value, actual_value, predicted_value in zip(
                        test_frame.loc[valid_test_mask, "Time"],
                        y_test_eval,
                        y_pred,
                    ):
                        source_time = pd.Timestamp(time_value)
                        all_predictions.append(
                            {
                                "TrainDays": train_days,
                                "ValDays": resolved_val_days,
                                "TestH": optuna_config.test_h,
                                "Date": current_day.strftime("%Y-%m-%d"),
                                "Season": season_name,
                                "Time": source_time,
                                "TargetTime": source_time + forecast_delta,
                                "Target": target,
                                "Model": model_name,
                                "actual": float(actual_value),
                                "predicted": float(predicted_value),
                            }
                        )
                    if shap_record_sink is not None or shap_iteration_exporter is not None:
                        iteration_shap_df = compute_optuna_iteration_shap_records(
                            model=final_model,
                            x_train=x_final_fit,
                            x_test=x_test_final,
                            train_days=train_days,
                            val_days=resolved_val_days,
                            test_h=optuna_config.test_h,
                            season_name=season_name,
                            current_day=current_day,
                            target=target,
                            model_name=model_name,
                            config=config,
                        )
                        if shap_record_sink is not None and not iteration_shap_df.empty:
                            shap_record_sink.extend(iteration_shap_df.to_dict("records"))
                        if shap_iteration_exporter is not None:
                            shap_iteration_exporter(
                                iteration_shap_df,
                                {
                                    "train_days": train_days,
                                    "val_days": resolved_val_days,
                                    "test_h": optuna_config.test_h,
                                    "season_name": season_name,
                                    "current_day": current_day,
                                    "target": target,
                                    "model_name": model_name,
                                },
                            )
                    if iteration_exporter is not None:
                        iteration_exporter(all_metrics, all_trials, all_predictions)
                    progress.advance(
                        detail=(
                            f"{current_day:%Y-%m-%d} train={train_days}d "
                            f"target={target} model={model_name}"
                        ),
                    )
        verbose_print(
            f"Completed Optuna {train_days}d walk-forward window: added {len(all_metrics) - metrics_before_window} metric rows.",
            verbose=effective_verbose,
            level=1,
        )

    metrics_df = pd.DataFrame(all_metrics)
    trials_df = pd.DataFrame(all_trials)
    preds_df = pd.DataFrame(all_predictions)
    if not metrics_df.empty:
        metrics_df["Date"] = pd.to_datetime(metrics_df["Date"])
    if not trials_df.empty:
        trials_df["Date"] = pd.to_datetime(trials_df["Date"])
    verbose_print(
        f"Optuna walk-forward modeling finished: metrics={len(metrics_df)}, trials={len(trials_df)}, "
        f"predictions={len(preds_df)}.",
        verbose=effective_verbose,
        level=1,
    )
    return metrics_df, trials_df, preds_df


def compute_physical_metrics(
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    verbose: int | None = None,
) -> pd.DataFrame:
    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    dataset_step = resolve_dataset_step(df_h, config=config)
    verbose_print(
        f"Physical baseline evaluation started for targets={config.targets}.",
        verbose=effective_verbose,
        level=1,
    )
    records: list[dict[str, object]] = []
    for season_name, current_day in iter_configured_daily_windows(config):
        test_start = _as_utc_timestamp(current_day)
        test_end = test_start + pd.Timedelta(days=1) - dataset_step
        day_frame = df_h[(df_h["Time"] >= test_start) & (df_h["Time"] <= test_end)].copy()
        if day_frame.empty:
            continue

        for target in config.targets:
            target_column = f"{target}_target"
            if target_column not in day_frame.columns:
                continue
            for model_name, prediction_column in [
                ("IRI", f"IRI_{target}_pred"),
                ("Аналит.", f"anal_{target}_pred"),
            ]:
                if prediction_column not in day_frame.columns:
                    continue
                valid = day_frame[[prediction_column, target_column]].dropna()
                if len(valid) < config.min_eval_rows:
                    continue
                records.append(
                    {
                        "Season": season_name,
                        "Date": pd.Timestamp(current_day),
                        "Target": target,
                        "Model": model_name,
                        "R2": round(float(r2_score(valid[target_column], valid[prediction_column])), 4),
                        "MAE": round(float(mean_absolute_error(valid[target_column], valid[prediction_column])), 4),
                        "TrainDays": config.plot_train_days,
                    }
                )
    phys_df = pd.DataFrame(records)
    verbose_print(
        f"Physical baseline evaluation finished: metrics={len(phys_df)}.",
        verbose=effective_verbose,
        level=1,
    )
    return phys_df


def build_summary_table(
    metrics_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    summary_frames = []
    if not metrics_df.empty:
        summary_frames.append(
            metrics_df[metrics_df["TrainDays"] == config.plot_train_days]
            .groupby(["Season", "Target", "Model"])[["R2", "MAE"]]
            .mean()
            .round(4)
            .reset_index()
        )
    if not phys_df.empty:
        summary_frames.append(
            phys_df.groupby(["Season", "Target", "Model"])[["R2", "MAE"]]
            .mean()
            .round(4)
            .reset_index()
        )
    if not summary_frames:
        return pd.DataFrame(columns=["Season", "Target", "Model", "R2", "MAE"])
    return pd.concat(summary_frames, ignore_index=True)


def build_optuna_summary_table(
    metrics_df: pd.DataFrame,
    phys_df: pd.DataFrame | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    summary_frames = []
    if not metrics_df.empty:
        ml_summary = (
            metrics_df[metrics_df["TrainDays"] == config.plot_train_days]
            .groupby(["TrainDays", "ValDays", "TestH", "Season", "Target", "Model"])[["R2", "MAE", "BestValScore"]]
            .mean()
            .round(4)
            .reset_index()
        )
        if not ml_summary.empty:
            summary_frames.append(ml_summary)
    if phys_df is not None and not phys_df.empty:
        phys_summary = (
            phys_df.groupby(["Season", "Target", "Model"])[["R2", "MAE"]]
            .mean()
            .round(4)
            .reset_index()
        )
        phys_summary.insert(0, "TestH", pd.NA)
        phys_summary.insert(0, "ValDays", pd.NA)
        phys_summary.insert(0, "TrainDays", config.plot_train_days)
        phys_summary["BestValScore"] = pd.NA
        phys_summary = phys_summary[
            ["TrainDays", "ValDays", "TestH", "Season", "Target", "Model", "R2", "MAE", "BestValScore"]
        ]
        summary_frames.append(phys_summary)
    if not summary_frames:
        return pd.DataFrame(
            columns=["TrainDays", "ValDays", "TestH", "Season", "Target", "Model", "R2", "MAE", "BestValScore"]
        )
    return pd.concat(summary_frames, ignore_index=True)


def build_metric_comparison_table(
    metrics_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    range_kind: str = "season",
    period_labels: Iterable[str] | None = None,
    date_start: date | datetime | str | pd.Timestamp | None = None,
    date_end: date | datetime | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    normalized_range_kind = range_kind.strip().lower()
    if normalized_range_kind not in {"season", "month", "week", "day"}:
        raise ValueError("range_kind must be one of: 'season', 'month', 'week', 'day'.")

    comparison_columns = ["Date", "Season", "Target", "Model", "R2", "MAE"]
    frames: list[pd.DataFrame] = []
    if not metrics_df.empty:
        ml_frame = metrics_df[metrics_df["TrainDays"] == config.plot_train_days].copy()
        if not ml_frame.empty:
            frames.append(ml_frame[comparison_columns])
    if not phys_df.empty:
        frames.append(phys_df[comparison_columns].copy())
    if not frames:
        return pd.DataFrame(
            columns=["RangeKind", "RangeLabel", "PeriodStart", "PeriodEnd", "Target", "Model", "R2", "MAE", "N_days"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["Date"]).copy()
    if combined.empty:
        return pd.DataFrame(
            columns=["RangeKind", "RangeLabel", "PeriodStart", "PeriodEnd", "Target", "Model", "R2", "MAE", "N_days"]
        )

    config_date_start, config_date_end = resolve_ml_date_range(config=config)
    resolved_date_start = config_date_start if date_start is None else _coerce_date_boundary(date_start, "date_start")
    resolved_date_end = config_date_end if date_end is None else _coerce_date_boundary(date_end, "date_end")

    if resolved_date_start is not None:
        combined = combined[combined["Date"].dt.date >= resolved_date_start].copy()
    if resolved_date_end is not None:
        combined = combined[combined["Date"].dt.date <= resolved_date_end].copy()
    if combined.empty:
        return pd.DataFrame(
            columns=["RangeKind", "RangeLabel", "PeriodStart", "PeriodEnd", "Target", "Model", "R2", "MAE", "N_days"]
        )

    if normalized_range_kind == "season":
        combined["RangeLabel"] = combined["Season"].fillna("Outside configured seasons")
    elif normalized_range_kind == "month":
        combined["RangeLabel"] = combined["Date"].dt.strftime("%Y-%m")
    elif normalized_range_kind == "week":
        iso_calendar = combined["Date"].dt.isocalendar()
        combined["RangeLabel"] = (
            iso_calendar["year"].astype(str) + "-W" + iso_calendar["week"].astype(str).str.zfill(2)
        )
    else:
        combined["RangeLabel"] = combined["Date"].dt.strftime("%Y-%m-%d")

    if period_labels is not None:
        selected_labels = {str(label) for label in period_labels}
        combined = combined[combined["RangeLabel"].isin(selected_labels)].copy()
    if combined.empty:
        return pd.DataFrame(
            columns=["RangeKind", "RangeLabel", "PeriodStart", "PeriodEnd", "Target", "Model", "R2", "MAE", "N_days"]
        )

    aggregated = (
        combined.groupby(["RangeLabel", "Target", "Model"], as_index=False)
        .agg(
            PeriodStart=("Date", "min"),
            PeriodEnd=("Date", "max"),
            R2=("R2", "mean"),
            MAE=("MAE", "mean"),
            N_days=("Date", "size"),
        )
        .sort_values(["PeriodStart", "Target", "Model"])
        .reset_index(drop=True)
    )
    aggregated.insert(0, "RangeKind", normalized_range_kind)
    return aggregated


def _to_excel_safe_datetime(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return timestamps


def _normalize_excel_name_suffix(excel_name_suffix: str | None) -> str | None:
    if excel_name_suffix is None:
        return None
    normalized = str(excel_name_suffix).strip()
    if not normalized:
        return None
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    if not normalized:
        raise ValueError("excel_name_suffix must contain at least one ASCII letter or digit.")
    return normalized


def build_excel_timestamp_token(exported_at: datetime | None = None) -> str:
    timestamp_source = datetime.now() if exported_at is None else exported_at
    return timestamp_source.strftime("%y%m%d_%H%M%S")


def build_excel_output_name(
    base_file_name: str,
    excel_name_suffix: str | None = None,
    excel_timestamp: str | None = None,
    file_name_prefix: str | None = None,
) -> str:
    normalized_suffix = _normalize_excel_name_suffix(excel_name_suffix)
    normalized_prefix = _normalize_excel_name_suffix(file_name_prefix)
    normalized_timestamp = _normalize_excel_name_suffix(
        build_excel_timestamp_token() if excel_timestamp is None else excel_timestamp
    )
    if normalized_timestamp is None:
        raise ValueError("excel_timestamp must contain at least one ASCII letter or digit.")

    file_path = Path(base_file_name)
    suffix = file_path.suffix or ".xlsx"
    stem = file_path.stem if file_path.suffix else file_path.name
    station_prefix = None
    remaining_suffix = normalized_suffix
    if normalized_suffix is not None:
        suffix_tokens = [token for token in normalized_suffix.split("_") if token]
        if suffix_tokens:
            maybe_station = suffix_tokens[-1].upper()
            if re.fullmatch(r"[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*", maybe_station):
                station_prefix = maybe_station
                remaining_tokens = suffix_tokens[:-1]
                remaining_suffix = "_".join(remaining_tokens) if remaining_tokens else None

    name_parts = []
    if station_prefix is not None:
        name_parts.append(station_prefix)
    if normalized_prefix is not None:
        name_parts.append(normalized_prefix)
    name_parts.append(stem)
    if remaining_suffix is not None:
        name_parts.append(remaining_suffix)
    name_parts.append(normalized_timestamp)
    return f"{'_'.join(name_parts)}{suffix}"


def build_partition_output_name(
    base_file_name: str,
    partition_name: str,
    excel_name_suffix: str | None = None,
    excel_timestamp: str | None = None,
    file_name_prefix: str | None = None,
) -> str:
    file_path = Path(base_file_name)
    suffix = file_path.suffix or ".csv"
    stem = file_path.stem if file_path.suffix else file_path.name
    return build_excel_output_name(
        f"{stem}_{partition_name}{suffix}",
        excel_name_suffix=excel_name_suffix,
        excel_timestamp=excel_timestamp,
        file_name_prefix=file_name_prefix,
    )


def export_dataframe_csv(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _resolve_station_excel_name_suffix(
    excel_name_suffix: str | None,
    station_code: str | None = None,
    *,
    ensure_unique: bool = False,
) -> str | None:
    normalized_station_code = station_code.strip().upper() if station_code else None
    raw_suffix = None if excel_name_suffix is None else str(excel_name_suffix).strip()

    if raw_suffix:
        resolved = raw_suffix.replace("{station_code}", normalized_station_code or "UNKNOWN")
        if normalized_station_code and "{station_code}" not in raw_suffix:
            station_pattern = rf"(^|[^A-Za-z0-9]){re.escape(normalized_station_code)}($|[^A-Za-z0-9])"
            if re.search(station_pattern, resolved.upper()) is None:
                resolved = f"{resolved}_{normalized_station_code}"
        return resolved
    if (ensure_unique or normalized_station_code) and normalized_station_code:
        return normalized_station_code
    return None


def resolve_results_output_dir(base_dir: str | Path, station_code: str | None) -> Path:
    raw_station_code = station_code.strip().upper() if station_code else "UNKNOWN"
    normalized_station_code = re.sub(r"[^A-Z0-9._-]+", "_", raw_station_code).strip("._-") or "UNKNOWN"
    return Path(base_dir) / "results" / normalized_station_code


def build_prediction_export_frame(
    preds_df: pd.DataFrame,
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    include_physical_baselines: bool = False,
) -> pd.DataFrame:
    base_columns = [
        "TrainDays",
        "RunDate",
        "Season",
        "SourceTime",
        "TargetTime",
        "Target",
        "Model",
        "LabelSource",
        "ActualLabel",
        "Predicted",
        "foF2_original",
        "foF2_filtered",
        "MUFD_original",
        "MUFD_filtered",
    ]
    export_frames: list[pd.DataFrame] = []
    if not preds_df.empty:
        export_frames.append(preds_df.copy())

    if include_physical_baselines:
        train_days_series = preds_df["TrainDays"] if "TrainDays" in preds_df.columns else pd.Series(dtype=float)
        baseline_train_days = sorted(
            {
                int(value)
                for value in pd.to_numeric(train_days_series, errors="coerce").dropna().tolist()
            }
        )
        if not baseline_train_days:
            baseline_train_days = [int(train_days) for train_days in config.window_list]
        dataset_step = resolve_dataset_step(df_h, config=config)
        forecast_delta = pd.Timedelta(hours=config.forecast_h)
        baseline_records: list[dict[str, object]] = []
        val_days_value = pd.NA
        test_h_value = pd.NA
        if "ValDays" in preds_df.columns:
            non_na_val_days = pd.to_numeric(preds_df["ValDays"], errors="coerce").dropna().unique()
            if len(non_na_val_days) == 1:
                val_days_value = int(non_na_val_days[0])
        if "TestH" in preds_df.columns:
            non_na_test_h = pd.to_numeric(preds_df["TestH"], errors="coerce").dropna().unique()
            if len(non_na_test_h) == 1:
                test_h_value = int(non_na_test_h[0])

        for season_name, current_day in iter_configured_daily_windows(config):
            test_start = _as_utc_timestamp(current_day)
            test_end = test_start + pd.Timedelta(days=1) - dataset_step
            day_frame = df_h[(df_h["Time"] >= test_start) & (df_h["Time"] <= test_end)].copy()
            if day_frame.empty:
                continue

            for target in config.targets:
                target_column = f"{target}_target"
                if target_column not in day_frame.columns:
                    continue
                for model_name, prediction_column in [
                    ("IRI", f"IRI_{target}_pred"),
                    ("Аналит.", f"anal_{target}_pred"),
                ]:
                    if prediction_column not in day_frame.columns:
                        continue
                    valid = day_frame[["Time", target_column, prediction_column]].dropna()
                    if valid.empty:
                        continue
                    source_times = pd.to_datetime(valid["Time"], utc=True, errors="coerce")
                    target_times = source_times + forecast_delta
                    for train_days in baseline_train_days:
                        for source_time, target_time, actual_value, predicted_value in zip(
                            source_times.tolist(),
                            target_times.tolist(),
                            pd.to_numeric(valid[target_column], errors="coerce").tolist(),
                            pd.to_numeric(valid[prediction_column], errors="coerce").tolist(),
                            strict=False,
                        ):
                            record = {
                                "TrainDays": train_days,
                                "Date": pd.Timestamp(current_day),
                                "Season": season_name,
                                "Time": source_time,
                                "TargetTime": target_time,
                                "Target": target,
                                "Model": model_name,
                                "actual": actual_value,
                                "predicted": predicted_value,
                            }
                            if "ValDays" in preds_df.columns:
                                record["ValDays"] = val_days_value
                            if "TestH" in preds_df.columns:
                                record["TestH"] = test_h_value
                            baseline_records.append(record)

        if baseline_records:
            export_frames.append(pd.DataFrame(baseline_records))

    if not export_frames:
        return pd.DataFrame(columns=base_columns)

    export_df = pd.concat(export_frames, ignore_index=True, sort=False)
    export_df["TrainDays"] = pd.to_numeric(export_df.get("TrainDays"), errors="coerce")
    export_df["SourceTime"] = _to_excel_safe_datetime(export_df["Time"])
    if "TargetTime" in export_df.columns:
        export_df["TargetTime"] = _to_excel_safe_datetime(export_df["TargetTime"])
    else:
        export_df["TargetTime"] = export_df["SourceTime"] + pd.Timedelta(hours=config.forecast_h)
    if "Date" in export_df.columns:
        export_df["RunDate"] = _to_excel_safe_datetime(export_df["Date"]).dt.normalize()
    else:
        export_df["RunDate"] = export_df["SourceTime"].dt.normalize()
    export_df["ActualLabel"] = pd.to_numeric(export_df.get("actual"), errors="coerce")
    export_df["Predicted"] = pd.to_numeric(export_df.get("predicted"), errors="coerce")

    if "Season" not in export_df.columns:
        export_df["Season"] = pd.NA
    if "Target" not in export_df.columns:
        export_df["Target"] = pd.NA
    if "Model" not in export_df.columns:
        export_df["Model"] = pd.NA

    label_source_map = {}
    for target in export_df["Target"].dropna().astype(str).unique():
        source_column = resolve_target_label_source_column(target, df_h, config=config)
        if source_column is None:
            label_source_map[target] = pd.NA
        else:
            label_source_map[target] = "filtered" if source_column.endswith("_savgol") else "raw"
    export_df["LabelSource"] = export_df["Target"].astype("string").map(label_source_map).astype("object")

    target_lookup = pd.DataFrame({"TargetTime": _to_excel_safe_datetime(df_h["Time"])})
    for source_column, export_column in (
        ("foF2", "foF2_original"),
        ("foF2_savgol", "foF2_filtered"),
        ("MUFD", "MUFD_original"),
        ("MUFD_savgol", "MUFD_filtered"),
    ):
        if source_column in df_h.columns:
            target_lookup[export_column] = pd.to_numeric(df_h[source_column], errors="coerce")
        else:
            target_lookup[export_column] = np.nan
    target_lookup = target_lookup.drop_duplicates(subset=["TargetTime"], keep="last")

    export_df = export_df.merge(target_lookup, on="TargetTime", how="left")
    optional_columns = [column for column in ("ValDays", "TestH") if column in export_df.columns]
    ordered_columns = base_columns[:1] + optional_columns + base_columns[1:]
    return export_df[ordered_columns].sort_values(
        ["TrainDays", "RunDate", "SourceTime", "Target", "Model"],
        kind="stable",
    ).reset_index(drop=True)


def export_prediction_results(
    preds_df: pd.DataFrame,
    df_h: pd.DataFrame,
    output_dir: str | Path,
    file_name: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    split_by_train_days: bool = True,
    *,
    include_physical_baselines: bool = False,
) -> None:
    output_path = Path(output_dir)
    export_df = build_prediction_export_frame(
        preds_df=preds_df,
        df_h=df_h,
        config=config,
        include_physical_baselines=include_physical_baselines,
    )
    export_dataframe_csv(export_df, output_path / file_name)
    if not split_by_train_days or export_df.empty or "TrainDays" not in export_df.columns:
        return

    file_path = Path(file_name)
    window_values = export_df["TrainDays"].dropna().tolist()
    for train_days in sorted({int(value) for value in window_values}):
        window_df = export_df[export_df["TrainDays"] == train_days].copy()
        window_file_name = f"{file_path.stem}_window_{train_days}d{file_path.suffix or '.csv'}"
        export_dataframe_csv(window_df, output_path / window_file_name)


def export_results(
    metrics_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    output_dir: str | Path,
    preds_df: pd.DataFrame | None = None,
    df_h: pd.DataFrame | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    excel_name_suffix: str | None = None,
    excel_timestamp: str | None = None,
    file_name_prefix: str | None = None,
    prediction_split_by_train_days: bool = True,
) -> None:
    output_path = Path(output_dir)
    resolved_timestamp = build_excel_timestamp_token() if excel_timestamp is None else excel_timestamp
    export_dataframe_csv(
        summary_df,
        output_path
        / build_partition_output_name(
            "metrics_all_models_2025_mac.csv",
            "summary",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    export_dataframe_csv(
        metrics_df,
        output_path
        / build_partition_output_name(
            "metrics_all_models_2025_mac.csv",
            "ML_daily",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    if not phys_df.empty:
        export_dataframe_csv(
            phys_df,
            output_path
            / build_partition_output_name(
                "metrics_all_models_2025_mac.csv",
                "phys_daily",
                excel_name_suffix,
                excel_timestamp=resolved_timestamp,
                file_name_prefix=file_name_prefix,
            ),
        )

    export_dataframe_csv(
        fi_df,
        output_path
        / build_partition_output_name(
            "feature_importance_2025_mac.csv",
            "daily_fi",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    if not fi_df.empty:
        season_mean_df = (
            fi_df.groupby(["TrainDays", "Season", "Target", "Model", "Feature"])["Importance"]
            .mean()
            .round(6)
            .reset_index()
        )
    else:
        season_mean_df = pd.DataFrame(
            columns=["TrainDays", "Season", "Target", "Model", "Feature", "Importance"]
        )
    export_dataframe_csv(
        season_mean_df,
        output_path
        / build_partition_output_name(
            "feature_importance_2025_mac.csv",
            "season_mean",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    if preds_df is not None and df_h is not None:
        export_prediction_results(
            preds_df=preds_df,
            df_h=df_h,
            output_dir=output_path,
            file_name=build_excel_output_name(
                "predicted_time_series_all_models_2025_mac.csv",
                excel_name_suffix,
                excel_timestamp=resolved_timestamp,
                file_name_prefix=file_name_prefix,
            ),
            config=config,
            split_by_train_days=prediction_split_by_train_days,
        )


def export_optuna_results(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    output_dir: str | Path,
    phys_df: pd.DataFrame | None = None,
    preds_df: pd.DataFrame | None = None,
    df_h: pd.DataFrame | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    excel_name_suffix: str | None = None,
    excel_timestamp: str | None = None,
    shap_df: pd.DataFrame | None = None,
    file_name_prefix: str | None = None,
    prediction_split_by_train_days: bool = True,
) -> None:
    output_path = Path(output_dir)
    resolved_timestamp = build_excel_timestamp_token() if excel_timestamp is None else excel_timestamp
    export_dataframe_csv(
        summary_df,
        output_path
        / build_partition_output_name(
            "metrics_optuna_models_2025_mac.csv",
            "summary",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    export_dataframe_csv(
        metrics_df,
        output_path
        / build_partition_output_name(
            "metrics_optuna_models_2025_mac.csv",
            "optuna_daily",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    export_dataframe_csv(
        trials_df,
        output_path
        / build_partition_output_name(
            "metrics_optuna_models_2025_mac.csv",
            "optuna_trials",
            excel_name_suffix,
            excel_timestamp=resolved_timestamp,
            file_name_prefix=file_name_prefix,
        ),
    )
    if phys_df is not None and not phys_df.empty:
        export_dataframe_csv(
            phys_df,
            output_path
            / build_partition_output_name(
                "metrics_optuna_models_2025_mac.csv",
                "phys_daily",
                excel_name_suffix,
                excel_timestamp=resolved_timestamp,
                file_name_prefix=file_name_prefix,
            ),
        )
    if preds_df is not None and df_h is not None:
        export_prediction_results(
            preds_df=preds_df,
            df_h=df_h,
            output_dir=output_path,
            file_name=build_excel_output_name(
                "predicted_time_series_optuna_models_2025_mac.csv",
                excel_name_suffix,
                excel_timestamp=resolved_timestamp,
                file_name_prefix=file_name_prefix,
            ),
            config=config,
            split_by_train_days=prediction_split_by_train_days,
            include_physical_baselines=True,
        )
    if shap_df is not None:
        export_dataframe_csv(
            shap_df,
            output_path
            / build_excel_output_name(
                "shap_optuna_models_2025_mac.csv",
                excel_name_suffix,
                excel_timestamp=resolved_timestamp,
                file_name_prefix=file_name_prefix,
            ),
        )


def export_optuna_iteration_shap_results(
    shap_df: pd.DataFrame,
    output_dir: str | Path,
    station_code: str | None = None,
    excel_name_suffix: str | None = None,
    excel_timestamp: str | None = None,
    file_name_prefix: str | None = None,
) -> None:
    if shap_df.empty:
        return

    first_row = shap_df.iloc[0]
    date_token = pd.Timestamp(first_row["Date"]).strftime("%Y%m%d")
    model_token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(first_row["Model"])).strip("._-") or "model"
    target_token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(first_row["Target"])).strip("._-") or "target"
    train_days = int(pd.to_numeric(first_row["TrainDays"], errors="coerce"))
    val_days = int(pd.to_numeric(first_row["ValDays"], errors="coerce"))
    test_h = int(pd.to_numeric(first_row["TestH"], errors="coerce"))
    iteration_dir = Path(output_dir) / "optuna_shap_iterations"
    file_name = build_excel_output_name(
        f"optuna_shap_{target_token}_{model_token}_{date_token}_{train_days}d_val{val_days}d_test{test_h}h.csv",
        excel_name_suffix=_resolve_station_excel_name_suffix(excel_name_suffix, station_code=station_code),
        excel_timestamp=excel_timestamp,
        file_name_prefix=file_name_prefix,
    )
    export_dataframe_csv(shap_df, iteration_dir / file_name)


def _safe_sample_indices(size: int, limit: int, random_state: np.random.Generator) -> np.ndarray:
    sample_size = min(size, limit)
    if sample_size <= 0:
        return np.array([], dtype=int)
    return random_state.choice(size, sample_size, replace=False)


def finalize_optuna_shap_frame(
    shap_records: list[dict[str, object]],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if not shap_records:
        return pd.DataFrame(columns=OPTUNA_SHAP_RESULT_COLUMNS)

    shap_df = pd.DataFrame(shap_records).copy()
    if shap_df.empty:
        return pd.DataFrame(columns=OPTUNA_SHAP_RESULT_COLUMNS)

    shap_df["Date"] = pd.to_datetime(shap_df["Date"], errors="coerce")
    shap_df = shap_df.sort_values(["TrainDays", "Date", "Target", "Model", "Feature"], kind="stable").reset_index(
        drop=True
    )
    shap_df["Pct_s"] = (
        shap_df.groupby(["TrainDays", "Target", "Model", "Feature"])["Pct"]
        .transform(lambda series: series.rolling(config.shap_smooth_days, center=True, min_periods=1).mean())
    )
    return shap_df.reindex(columns=OPTUNA_SHAP_RESULT_COLUMNS)


def _compute_xgboost_shap_values(model: object, explain_frame: pd.DataFrame) -> np.ndarray:
    if xgb is None:  # pragma: no cover - defensive guard
        raise RuntimeError("XGBoost is not installed.")
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(explain_frame, feature_names=list(explain_frame.columns))
    contribs = booster.predict(dmatrix, pred_contribs=True)
    contribs = np.asarray(contribs)
    if contribs.ndim == 1:
        contribs = contribs.reshape(1, -1)
    if contribs.shape[1] == len(explain_frame.columns) + 1:
        contribs = contribs[:, :-1]
    return contribs


def _compute_linear_model_shap_values(
    model: object,
    background: pd.DataFrame,
    explain_frame: pd.DataFrame,
    shap_module,
) -> np.ndarray:
    linear_model = model
    shap_background: pd.DataFrame | np.ndarray = background
    shap_explain: pd.DataFrame | np.ndarray = explain_frame

    if isinstance(model, Pipeline):
        if len(model.steps) < 2:
            raise RuntimeError("ElasticNet pipeline must include at least one transformer and a final estimator.")
        transformer = model[:-1]
        linear_model = model.steps[-1][1]
        shap_background = transformer.transform(background)
        shap_explain = transformer.transform(explain_frame)

    return shap_module.LinearExplainer(linear_model, shap_background).shap_values(shap_explain)


def compute_optuna_iteration_shap_records(
    model: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_days: int,
    val_days: int,
    test_h: int,
    season_name: str,
    current_day: datetime,
    target: str,
    model_name: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("shap is required to compute Optuna SHAP diagnostics.") from exc

    if x_train.empty or x_test.empty:
        return pd.DataFrame(columns=OPTUNA_SHAP_RESULT_COLUMNS)

    rng = np.random.default_rng(42)
    background_idx = _safe_sample_indices(len(x_train), config.shap_background, rng)
    explain_idx = _safe_sample_indices(len(x_test), config.shap_sample, rng)
    if len(background_idx) == 0 or len(explain_idx) == 0:
        return pd.DataFrame(columns=OPTUNA_SHAP_RESULT_COLUMNS)

    background = x_train.iloc[background_idx].copy()
    explain_frame = x_test.iloc[explain_idx].copy()

    if model_name == "XGBoost":
        shap_values = _compute_xgboost_shap_values(model, explain_frame)
    elif model_name == "RandomForest":
        shap_values = shap.TreeExplainer(model).shap_values(explain_frame)
    elif model_name == "ElasticNet":
        shap_values = _compute_linear_model_shap_values(model, background, explain_frame, shap)
    else:  # pragma: no cover - defensive fallback
        shap_values = shap.Explainer(model, background)(explain_frame).values

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if shap_values.ndim > 2:
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    records = [
        {
            "TrainDays": train_days,
            "ValDays": val_days,
            "TestH": test_h,
            "Date": pd.Timestamp(current_day),
            "Season": season_name,
            "Target": target,
            "Model": model_name,
            "Feature": feature,
            "MeanAbsShap": float(mean_abs[index]),
            "Pct": float(mean_abs[index] / total * 100.0),
            "NBackground": int(len(background)),
            "NExplain": int(len(explain_frame)),
        }
        for index, feature in enumerate(explain_frame.columns)
    ]
    return pd.DataFrame(records).reindex(columns=OPTUNA_SHAP_RESULT_COLUMNS[:-1])


def compute_shap_records(
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    verbose: int | None = None,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("shap is required to compute SHAP diagnostics.") from exc

    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    rng = np.random.default_rng(42)
    feature_pool = get_feature_pool(df_h)
    records: list[dict[str, object]] = []
    verbose_print(
        f"SHAP diagnostics started for targets={config.targets}.",
        verbose=effective_verbose,
        level=1,
    )

    phys_cfg: dict[str, dict[str, tuple[Callable[[np.ndarray], np.ndarray], list[str]]]] = {
        "foF2": {
            "IRI": (lambda data: data[:, 0], ["IRI_foF2_pred"]),
            "Аналит.": (
                lambda data: 0.7 * np.sqrt(np.maximum(data[:, 0] * 1e16 / (config.tau_km * 1e3 * 1.24e10), 0))
                + 0.3 * data[:, 1] * 2.3,
                ["TEC", "foE"],
            ),
        },
        "MUFD": {
            "IRI": (lambda data: data[:, 0] * data[:, 1], ["IRI_M3000_pred", "IRI_foF2_pred"]),
            "Аналит.": (
                lambda data: np.where(data[:, 0] + 176 > 0, 1490 / (data[:, 0] + 176), np.nan)
                * np.sqrt(np.maximum(data[:, 1] * 1e16 / (config.tau_km * 1e3 * 1.24e10), 0)),
                ["hmF2", "TEC"],
            ),
        },
    }

    for target in config.targets:
        target_column = f"{target}_target"
        for season_name, current_day in iter_configured_daily_windows(config):
            del season_name
            train_frame, test_frame, _ = split_walk_forward_window(
                df_h=df_h,
                current_day=current_day,
                train_days=config.plot_train_days,
                forecast_h=config.forecast_h,
            )
            if train_frame.empty or test_frame.empty or target_column not in train_frame.columns:
                continue

            ml_features = [
                feature
                for feature in select_features(train_frame, feature_pool, config.min_feature_coverage)
                if feature in config.shap_ml_features
            ]
            y_train = train_frame[target_column]
            valid_train = y_train.notna()

            if ml_features and valid_train.sum() >= config.min_train_target_rows:
                imputer = SimpleImputer(strategy="median")
                train_imputed = pd.DataFrame(
                    imputer.fit_transform(train_frame.loc[valid_train, ml_features]),
                    columns=ml_features,
                )
                test_imputed = pd.DataFrame(
                    imputer.transform(test_frame[ml_features]),
                    columns=ml_features,
                )
                if not test_imputed.empty:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_leaf=3,
                        n_jobs=config.model_n_jobs,
                        random_state=42,
                    )
                    model.fit(train_imputed, y_train.loc[valid_train])
                    sample_idx = _safe_sample_indices(len(test_imputed), config.shap_sample, rng)
                    if len(sample_idx) > 0:
                        shap_values = shap.TreeExplainer(model).shap_values(test_imputed.iloc[sample_idx])
                        mean_abs = np.abs(shap_values).mean(axis=0)
                        total = mean_abs.sum() or 1.0
                        for index, feature in enumerate(ml_features):
                            records.append(
                                {
                                    "Date": pd.Timestamp(current_day),
                                    "Target": target,
                                    "Model": "ML",
                                    "Feature": feature,
                                    "Pct": mean_abs[index] / total * 100,
                                }
                            )

            for model_name, (predict_fn, feature_columns) in phys_cfg[target].items():
                if not all(column in test_frame.columns for column in feature_columns + [target_column]):
                    continue
                subset = test_frame[feature_columns + [target_column]].dropna()
                if len(subset) < 10:
                    continue
                x_physical = subset[feature_columns].to_numpy()
                background_idx = _safe_sample_indices(len(x_physical), config.shap_background, rng)
                explain_idx = _safe_sample_indices(len(x_physical), config.shap_sample, rng)
                if len(background_idx) == 0 or len(explain_idx) == 0:
                    continue
                explainer = shap.KernelExplainer(predict_fn, x_physical[background_idx])
                shap_values = explainer.shap_values(
                    x_physical[explain_idx],
                    nsamples=config.shap_kernel_samples,
                    silent=True,
                )
                mean_abs = np.abs(shap_values).mean(axis=0)
                total = mean_abs.sum() or 1.0
                for index, feature in enumerate(feature_columns):
                    records.append(
                        {
                            "Date": pd.Timestamp(current_day),
                            "Target": target,
                            "Model": model_name,
                            "Feature": feature,
                            "Pct": mean_abs[index] / total * 100,
                        }
                    )

    shap_df = pd.DataFrame(records)
    if shap_df.empty:
        verbose_print(
            "SHAP diagnostics finished: no records were generated.",
            verbose=effective_verbose,
            level=1,
        )
        return pd.DataFrame(columns=SHAP_RESULT_COLUMNS)
    shap_df = shap_df.sort_values("Date").reset_index(drop=True)
    shap_df["Pct_s"] = (
        shap_df.groupby(["Target", "Model", "Feature"])["Pct"]
        .transform(lambda series: series.rolling(config.shap_smooth_days, center=True, min_periods=1).mean())
    )
    verbose_print(
        f"SHAP diagnostics finished: records={len(shap_df)}.",
        verbose=effective_verbose,
        level=1,
    )
    return shap_df


def _run_walk_forward_window_task(
    df_h: pd.DataFrame,
    train_days: int,
    config: PipelineConfig,
    output_dir: str | Path,
    export_excel: bool,
    verbose: int | None,
    excel_name_suffix: str | None,
    excel_timestamp: str | None,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_config = replace(config, window_list=(train_days,), plot_train_days=train_days)
    effective_verbose = resolve_verbose(config=window_config, verbose=verbose)
    file_name_prefix = build_window_file_prefix(train_days)

    iteration_exporter = None
    if export_excel:
        def iteration_exporter(
            all_metrics: list[dict[str, object]],
            all_importance: list[dict[str, object]],
            all_predictions: list[dict[str, object]],
        ) -> None:
            metrics_df = pd.DataFrame(all_metrics)
            fi_df = pd.DataFrame(all_importance)
            preds_df = pd.DataFrame(all_predictions)
            export_results(
                metrics_df=metrics_df,
                phys_df=pd.DataFrame(),
                summary_df=build_summary_table(metrics_df, pd.DataFrame(), config=window_config),
                fi_df=fi_df,
                output_dir=output_dir,
                preds_df=preds_df,
                df_h=df_h,
                config=window_config,
                excel_name_suffix=excel_name_suffix,
                excel_timestamp=excel_timestamp,
                file_name_prefix=file_name_prefix,
                prediction_split_by_train_days=False,
            )

    metrics_df, fi_df, preds_df = run_walk_forward_models(
        df_h=df_h,
        config=window_config,
        verbose=effective_verbose,
        iteration_exporter=iteration_exporter,
    )
    if export_excel:
        export_results(
            metrics_df=metrics_df,
            phys_df=pd.DataFrame(),
            summary_df=build_summary_table(metrics_df, pd.DataFrame(), config=window_config),
            fi_df=fi_df,
            output_dir=output_dir,
            preds_df=preds_df,
            df_h=df_h,
            config=window_config,
            excel_name_suffix=excel_name_suffix,
            excel_timestamp=excel_timestamp,
            file_name_prefix=file_name_prefix,
            prediction_split_by_train_days=False,
        )
    return train_days, metrics_df, fi_df, preds_df


def _run_optuna_window_task(
    df_h: pd.DataFrame,
    train_days: int,
    config: PipelineConfig,
    optuna_config: OptunaConfig,
    output_dir: str | Path,
    export_excel: bool,
    compute_shap: bool,
    verbose: int | None,
    station_code: str | None,
    excel_name_suffix: str | None,
    excel_timestamp: str | None,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_config = replace(config, window_list=(train_days,), plot_train_days=train_days)
    effective_verbose = resolve_verbose(config=window_config, verbose=verbose)
    file_name_prefix = build_window_file_prefix(train_days)
    shap_records: list[dict[str, object]] | None = [] if compute_shap else None

    iteration_exporter = None
    if export_excel:
        def iteration_exporter(
            all_metrics: list[dict[str, object]],
            all_trials: list[dict[str, object]],
            all_predictions: list[dict[str, object]],
        ) -> None:
            metrics_df = pd.DataFrame(all_metrics)
            export_optuna_results(
            metrics_df=metrics_df,
            summary_df=build_optuna_summary_table(metrics_df, config=window_config),
            trials_df=pd.DataFrame(all_trials),
            output_dir=output_dir,
            preds_df=pd.DataFrame(all_predictions),
                df_h=df_h,
                config=window_config,
                excel_name_suffix=excel_name_suffix,
                excel_timestamp=excel_timestamp,
                shap_df=finalize_optuna_shap_frame(shap_records, config=window_config) if shap_records is not None else None,
                file_name_prefix=file_name_prefix,
                prediction_split_by_train_days=False,
            )

    shap_iteration_exporter = None
    if export_excel and compute_shap:
        def shap_iteration_exporter(iteration_shap_df: pd.DataFrame, _: dict[str, object]) -> None:
            export_optuna_iteration_shap_results(
                iteration_shap_df,
                output_dir=output_dir,
                station_code=station_code,
                excel_name_suffix=excel_name_suffix,
                excel_timestamp=excel_timestamp,
                file_name_prefix=file_name_prefix,
            )

    metrics_df, trials_df, preds_df = run_optuna_walk_forward_models(
        df_h=df_h,
        config=window_config,
        optuna_config=optuna_config,
        verbose=effective_verbose,
        iteration_exporter=iteration_exporter,
        shap_record_sink=shap_records,
        shap_iteration_exporter=shap_iteration_exporter,
    )
    shap_df = finalize_optuna_shap_frame(shap_records, config=window_config) if shap_records is not None else pd.DataFrame(
        columns=OPTUNA_SHAP_RESULT_COLUMNS
    )
    if export_excel:
        export_optuna_results(
            metrics_df=metrics_df,
            summary_df=build_optuna_summary_table(metrics_df, config=window_config),
            trials_df=trials_df,
            output_dir=output_dir,
            preds_df=preds_df,
            df_h=df_h,
            config=window_config,
            excel_name_suffix=excel_name_suffix,
            excel_timestamp=excel_timestamp,
            shap_df=shap_df,
            file_name_prefix=file_name_prefix,
            prediction_split_by_train_days=False,
        )
    return train_days, metrics_df, trials_df, preds_df, shap_df


def run_pipeline(
    base_dir: str | Path = ".",
    datasets_dir: str = "datasets",
    station_code: str | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    iri_config: IRIConfig = DEFAULT_IRI_CONFIG,
    export_excel: bool = True,
    compute_shap: bool = True,
    verbose: int | None = None,
    excel_name_suffix: str | None = None,
) -> dict[str, pd.DataFrame | list[str] | PipelineConfig | IRIConfig]:
    resolved_window_workers, resolved_model_n_jobs = resolve_window_parallel_settings(config)
    effective_config = replace(config, model_n_jobs=resolved_model_n_jobs)
    effective_verbose = resolve_verbose(config=effective_config, verbose=verbose)
    verbose_print(
        f"Pipeline started: base_dir={Path(base_dir).resolve()}, datasets_dir={datasets_dir}, station_code={station_code}.",
        verbose=effective_verbose,
        level=1,
    )
    verbose_print(
        f"Window execution settings: windows={effective_config.window_list}, workers={resolved_window_workers}, "
        f"model_n_jobs={resolved_model_n_jobs}.",
        verbose=effective_verbose,
        level=1,
    )
    source_paths, dataset_metadata = resolve_dataset_source_paths(
        base_dir=base_dir,
        datasets_dir=datasets_dir,
        station_code=station_code,
    )
    verbose_print(
        f"Resolved station {dataset_metadata.get('station_code')} ({dataset_metadata.get('station_name')}) "
        f"for years={dataset_metadata.get('available_years')}.",
        verbose=effective_verbose,
        level=1,
    )
    df_raw = load_giro_dataset(source_paths, min_cs=effective_config.min_cs, verbose=effective_verbose)
    station_metadata = dict(df_raw.attrs.get("station_metadata", {}))
    for key, value in dataset_metadata.items():
        station_metadata.setdefault(key, value)
    resolved_export_suffix = _resolve_station_excel_name_suffix(
        excel_name_suffix,
        station_code=station_metadata.get("station_code") or station_code,
    )
    results_output_dir = resolve_results_output_dir(base_dir, station_metadata.get("station_code") or station_code)
    export_timestamp = build_excel_timestamp_token() if export_excel else None
    iri_config = resolve_iri_config_from_metadata(
        station_metadata,
        iri_config=iri_config,
        verbose=effective_verbose,
    )
    df_daily = build_daily_frame(df_raw, config=effective_config)
    df_h = build_dataset_frame(df_raw, config=effective_config, verbose=effective_verbose)
    dataset_step = resolve_dataset_step(df_h, config=effective_config)
    verbose_print(
        f"Time frames built: daily_rows={len(df_daily)}, dataset_rows={len(df_h)}, dataset_step={dataset_step}.",
        verbose=effective_verbose,
        level=1,
    )
    df_h = add_iri_baseline(df_h, config=effective_config, iri_config=iri_config)
    verbose_print("IRI baseline added.", verbose=effective_verbose, level=1)
    df_h = add_analytic_baseline(df_h, config=effective_config)
    verbose_print("Analytic baseline added.", verbose=effective_verbose, level=1)
    savgol_diagnostics_df = pd.DataFrame(
        df_h.attrs.get("savgol_diagnostics_records", []),
        columns=SAVGOL_DIAGNOSTIC_COLUMNS,
    )
    savgol_metadata = dict(df_h.attrs.get("savgol_metadata", {}))
    station_metadata["savgol_filter"] = savgol_metadata

    if len(effective_config.window_list) > 1:
        window_results = Parallel(n_jobs=resolved_window_workers, backend="loky")(
            delayed(_run_walk_forward_window_task)(
                df_h,
                train_days,
                effective_config,
                results_output_dir,
                export_excel,
                verbose,
                resolved_export_suffix,
                export_timestamp,
            )
            for train_days in effective_config.window_list
        )
        metrics_frames = [result[1] for result in window_results if not result[1].empty]
        fi_frames = [result[2] for result in window_results if not result[2].empty]
        preds_frames = [result[3] for result in window_results if not result[3].empty]
        metrics_df = (
            pd.concat(metrics_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Target", "Model"], kind="stable")
            .reset_index(drop=True)
            if metrics_frames
            else pd.DataFrame()
        )
        fi_df = (
            pd.concat(fi_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Target", "Model", "Feature"], kind="stable")
            .reset_index(drop=True)
            if fi_frames
            else pd.DataFrame()
        )
        preds_df = (
            pd.concat(preds_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Time", "Target", "Model"], kind="stable")
            .reset_index(drop=True)
            if preds_frames
            else pd.DataFrame()
        )
    else:
        iteration_exporter = None
        if export_excel:
            def iteration_exporter(
                all_metrics: list[dict[str, object]],
                all_importance: list[dict[str, object]],
                all_predictions: list[dict[str, object]],
            ) -> None:
                metrics_df = pd.DataFrame(all_metrics)
                fi_df = pd.DataFrame(all_importance)
                preds_df = pd.DataFrame(all_predictions)
                export_results(
                    metrics_df=metrics_df,
                    phys_df=pd.DataFrame(),
                    summary_df=build_summary_table(metrics_df, pd.DataFrame(), config=effective_config),
                    fi_df=fi_df,
                    output_dir=results_output_dir,
                    preds_df=preds_df,
                    df_h=df_h,
                    config=effective_config,
                    excel_name_suffix=resolved_export_suffix,
                    excel_timestamp=export_timestamp,
                )

        metrics_df, fi_df, preds_df = run_walk_forward_models(
            df_h,
            config=effective_config,
            verbose=effective_verbose,
            iteration_exporter=iteration_exporter,
        )
    phys_df = compute_physical_metrics(df_h, config=effective_config, verbose=effective_verbose)
    summary_df = build_summary_table(metrics_df, phys_df, config=effective_config)
    verbose_print(f"Summary table built: rows={len(summary_df)}.", verbose=effective_verbose, level=1)
    if compute_shap:
        shap_df = compute_shap_records(df_h, config=effective_config, verbose=effective_verbose)
    else:
        verbose_print("Skipping SHAP diagnostics.", verbose=effective_verbose, level=1)
        shap_df = pd.DataFrame(columns=SHAP_RESULT_COLUMNS)

    if export_excel:
        verbose_print(
            f"Exporting CSV results to {results_output_dir.resolve()}.",
            verbose=effective_verbose,
            level=1,
        )
        export_results(
            metrics_df=metrics_df,
            phys_df=phys_df,
            summary_df=summary_df,
            fi_df=fi_df,
            output_dir=results_output_dir,
            preds_df=preds_df,
            df_h=df_h,
            config=effective_config,
            excel_name_suffix=resolved_export_suffix,
            excel_timestamp=export_timestamp,
        )
    verbose_print("Pipeline finished successfully.", verbose=effective_verbose, level=1)

    return {
        "config": effective_config,
        "iri_config": iri_config,
        "df_raw": df_raw,
        "df_daily": df_daily,
        "df_h": df_h,
        "savgol_diagnostics_df": savgol_diagnostics_df,
        "savgol_metadata": savgol_metadata,
        "metrics_df": metrics_df,
        "fi_df": fi_df,
        "preds_df": preds_df,
        "phys_df": phys_df,
        "summary_df": summary_df,
        "shap_df": shap_df,
        "feature_pool": get_feature_pool(df_h),
        "station_metadata": station_metadata,
        "source_paths": source_paths,
        "export_dir": str(results_output_dir.resolve()),
    }


def compute_physical_shap_records_for_optuna(
    df_h: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    verbose: int | None = None,
) -> pd.DataFrame:
    """Compute SHAP records for IRI and Аналит. physical models across all configured windows.

    Returns a DataFrame formatted as OPTUNA_SHAP_RESULT_COLUMNS so it can be
    concatenated with the ML shap_df produced by run_optuna_pipeline.
    """
    try:
        import shap as shap_module
    except ImportError as exc:
        raise ImportError("shap is required to compute physical SHAP diagnostics.") from exc

    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    rng = np.random.default_rng(42)

    phys_cfg: dict[str, dict[str, tuple]] = {
        "foF2": {
            "IRI": (lambda data: data[:, 0], ["IRI_foF2_pred"]),
            "Аналит.": (
                lambda data: 0.7 * np.sqrt(np.maximum(data[:, 0] * 1e16 / (config.tau_km * 1e3 * 1.24e10), 0))
                + 0.3 * data[:, 1] * 2.3,
                ["TEC", "foE"],
            ),
        },
        "MUFD": {
            "IRI": (lambda data: data[:, 0] * data[:, 1], ["IRI_M3000_pred", "IRI_foF2_pred"]),
            "Аналит.": (
                lambda data: np.where(data[:, 0] + 176 > 0, 1490 / (data[:, 0] + 176), np.nan)
                * np.sqrt(np.maximum(data[:, 1] * 1e16 / (config.tau_km * 1e3 * 1.24e10), 0)),
                ["hmF2", "TEC"],
            ),
        },
    }

    records: list[dict[str, object]] = []
    verbose_print(
        f"Physical SHAP (IRI + Аналит.) started: windows={config.window_list}, targets={config.targets}.",
        verbose=effective_verbose,
        level=1,
    )

    for train_days in config.window_list:
        verbose_print(
            f"  Physical SHAP: train_days={train_days}.",
            verbose=effective_verbose,
            level=1,
        )
        for target in config.targets:
            target_column = f"{target}_target"
            for season_name, current_day in iter_configured_daily_windows(config):
                _, test_frame, _ = split_walk_forward_window(
                    df_h=df_h,
                    current_day=current_day,
                    train_days=train_days,
                    forecast_h=config.forecast_h,
                )
                if test_frame.empty or target_column not in test_frame.columns:
                    continue

                for model_name, (predict_fn, feature_columns) in phys_cfg[target].items():
                    needed = feature_columns + [target_column]
                    if not all(col in test_frame.columns for col in needed):
                        continue
                    subset = test_frame[needed].dropna()
                    if len(subset) < 10:
                        continue
                    x_physical = subset[feature_columns].to_numpy()
                    background_idx = _safe_sample_indices(len(x_physical), config.shap_background, rng)
                    explain_idx = _safe_sample_indices(len(x_physical), config.shap_sample, rng)
                    if len(background_idx) == 0 or len(explain_idx) == 0:
                        continue
                    try:
                        explainer = shap_module.KernelExplainer(predict_fn, x_physical[background_idx])
                        shap_values = explainer.shap_values(
                            x_physical[explain_idx],
                            nsamples=config.shap_kernel_samples,
                            silent=True,
                        )
                    except Exception:
                        continue
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    total = mean_abs.sum() or 1.0
                    for index, feature in enumerate(feature_columns):
                        records.append(
                            {
                                "TrainDays": train_days,
                                "ValDays": 1,
                                "TestH": config.forecast_h,
                                "Date": pd.Timestamp(current_day),
                                "Season": season_name,
                                "Target": target,
                                "Model": model_name,
                                "Feature": feature,
                                "MeanAbsShap": float(mean_abs[index]),
                                "Pct": float(mean_abs[index] / total * 100),
                                "NBackground": len(background_idx),
                                "NExplain": len(explain_idx),
                            }
                        )

    if not records:
        verbose_print(
            "Physical SHAP finished: no records generated.",
            verbose=effective_verbose,
            level=1,
        )
        return pd.DataFrame(columns=OPTUNA_SHAP_RESULT_COLUMNS)

    phys_shap_df = pd.DataFrame(records)
    phys_shap_df["Date"] = pd.to_datetime(phys_shap_df["Date"], errors="coerce")
    phys_shap_df = phys_shap_df.sort_values(
        ["TrainDays", "Date", "Target", "Model", "Feature"], kind="stable"
    ).reset_index(drop=True)
    phys_shap_df["Pct_s"] = (
        phys_shap_df.groupby(["TrainDays", "Target", "Model", "Feature"])["Pct"]
        .transform(lambda series: series.rolling(config.shap_smooth_days, center=True, min_periods=1).mean())
    )
    verbose_print(
        f"Physical SHAP finished: records={len(phys_shap_df)}.",
        verbose=effective_verbose,
        level=1,
    )
    return phys_shap_df.reindex(columns=OPTUNA_SHAP_RESULT_COLUMNS)


def run_optuna_pipeline(
    base_dir: str | Path = ".",
    datasets_dir: str = "datasets",
    station_code: str | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG,
    export_excel: bool = True,
    compute_shap: bool = True,
    verbose: int | None = None,
    excel_name_suffix: str | None = None,
) -> dict[str, pd.DataFrame | list[str] | PipelineConfig | OptunaConfig | IRIConfig]:
    resolved_window_workers, resolved_model_n_jobs = resolve_window_parallel_settings(config)
    effective_config = replace(config, model_n_jobs=resolved_model_n_jobs)
    effective_verbose = resolve_verbose(config=effective_config, verbose=verbose)
    verbose_print(
        "Optuna pipeline started: "
        f"base_dir={Path(base_dir).resolve()}, datasets_dir={datasets_dir}, station_code={station_code}.",
        verbose=effective_verbose,
        level=1,
    )
    verbose_print(
        f"Window execution settings: windows={effective_config.window_list}, workers={resolved_window_workers}, "
        f"model_n_jobs={resolved_model_n_jobs}.",
        verbose=effective_verbose,
        level=1,
    )
    source_paths, dataset_metadata = resolve_dataset_source_paths(
        base_dir=base_dir,
        datasets_dir=datasets_dir,
        station_code=station_code,
    )
    verbose_print(
        f"Resolved station {dataset_metadata.get('station_code')} ({dataset_metadata.get('station_name')}) "
        f"for years={dataset_metadata.get('available_years')}.",
        verbose=effective_verbose,
        level=1,
    )
    df_raw = load_giro_dataset(source_paths, min_cs=effective_config.min_cs, verbose=effective_verbose)
    station_metadata = dict(df_raw.attrs.get("station_metadata", {}))
    for key, value in dataset_metadata.items():
        station_metadata.setdefault(key, value)
    resolved_export_suffix = _resolve_station_excel_name_suffix(
        excel_name_suffix,
        station_code=station_metadata.get("station_code") or station_code,
    )
    results_output_dir = resolve_results_output_dir(base_dir, station_metadata.get("station_code") or station_code)
    export_timestamp = build_excel_timestamp_token() if export_excel else None
    iri_config = resolve_iri_config_from_metadata(
        station_metadata,
        iri_config=DEFAULT_IRI_CONFIG,
        verbose=effective_verbose,
    )

    df_daily = build_daily_frame(df_raw, config=effective_config)
    df_h = build_dataset_frame(df_raw, config=effective_config, verbose=effective_verbose)
    dataset_step = resolve_dataset_step(df_h, config=effective_config)
    verbose_print(
        f"Time frames built for Optuna: daily_rows={len(df_daily)}, dataset_rows={len(df_h)}, dataset_step={dataset_step}.",
        verbose=effective_verbose,
        level=1,
    )
    df_h = add_iri_baseline(df_h, config=effective_config, iri_config=iri_config)
    verbose_print("IRI baseline added for Optuna.", verbose=effective_verbose, level=1)
    df_h = add_analytic_baseline(df_h, config=effective_config)
    verbose_print("Analytic baseline added for Optuna.", verbose=effective_verbose, level=1)
    savgol_diagnostics_df = pd.DataFrame(
        df_h.attrs.get("savgol_diagnostics_records", []),
        columns=SAVGOL_DIAGNOSTIC_COLUMNS,
    )
    savgol_metadata = dict(df_h.attrs.get("savgol_metadata", {}))
    station_metadata["savgol_filter"] = savgol_metadata

    if len(effective_config.window_list) > 1:
        window_results = Parallel(n_jobs=resolved_window_workers, backend="loky")(
            delayed(_run_optuna_window_task)(
                df_h,
                train_days,
                effective_config,
                optuna_config,
                results_output_dir,
                export_excel,
                compute_shap,
                verbose,
                station_metadata.get("station_code") or station_code,
                resolved_export_suffix,
                export_timestamp,
            )
            for train_days in effective_config.window_list
        )
        metrics_frames = [result[1] for result in window_results if not result[1].empty]
        trials_frames = [result[2] for result in window_results if not result[2].empty]
        preds_frames = [result[3] for result in window_results if not result[3].empty]
        shap_frames = [result[4] for result in window_results if not result[4].empty]
        metrics_df = (
            pd.concat(metrics_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Target", "Model"], kind="stable")
            .reset_index(drop=True)
            if metrics_frames
            else pd.DataFrame()
        )
        trials_df = (
            pd.concat(trials_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Target", "Model", "TrialNumber"], kind="stable")
            .reset_index(drop=True)
            if trials_frames
            else pd.DataFrame()
        )
        preds_df = (
            pd.concat(preds_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Time", "Target", "Model"], kind="stable")
            .reset_index(drop=True)
            if preds_frames
            else pd.DataFrame()
        )
        shap_df = (
            pd.concat(shap_frames, ignore_index=True)
            .sort_values(["TrainDays", "Date", "Target", "Model", "Feature"], kind="stable")
            .reset_index(drop=True)
            if shap_frames
            else pd.DataFrame(columns=OPTUNA_SHAP_RESULT_COLUMNS)
        )
    else:
        shap_records: list[dict[str, object]] | None = [] if compute_shap else None
        iteration_exporter = None
        if export_excel:
            def iteration_exporter(
                all_metrics: list[dict[str, object]],
                all_trials: list[dict[str, object]],
                all_predictions: list[dict[str, object]],
            ) -> None:
                metrics_df = pd.DataFrame(all_metrics)
                export_optuna_results(
                    metrics_df=metrics_df,
                    summary_df=build_optuna_summary_table(metrics_df, config=effective_config),
                    trials_df=pd.DataFrame(all_trials),
                    output_dir=results_output_dir,
                    preds_df=pd.DataFrame(all_predictions),
                    df_h=df_h,
                    config=effective_config,
                    excel_name_suffix=resolved_export_suffix,
                    excel_timestamp=export_timestamp,
                    shap_df=finalize_optuna_shap_frame(shap_records, config=effective_config) if shap_records is not None else None,
                )

        shap_iteration_exporter = None
        if export_excel and compute_shap:
            def shap_iteration_exporter(iteration_shap_df: pd.DataFrame, _: dict[str, object]) -> None:
                export_optuna_iteration_shap_results(
                    iteration_shap_df,
                    output_dir=results_output_dir,
                    station_code=station_metadata.get("station_code") or station_code,
                    excel_name_suffix=excel_name_suffix,
                    excel_timestamp=export_timestamp,
                )

        metrics_df, trials_df, preds_df = run_optuna_walk_forward_models(
            df_h=df_h,
            config=effective_config,
            optuna_config=optuna_config,
            verbose=effective_verbose,
            iteration_exporter=iteration_exporter,
            shap_record_sink=shap_records,
            shap_iteration_exporter=shap_iteration_exporter,
        )
        shap_df = finalize_optuna_shap_frame(shap_records, config=effective_config) if shap_records is not None else pd.DataFrame(
            columns=OPTUNA_SHAP_RESULT_COLUMNS
        )
    if compute_shap:
        phys_shap_df = compute_physical_shap_records_for_optuna(
            df_h, config=effective_config, verbose=effective_verbose
        )
        if not phys_shap_df.empty:
            shap_df = pd.concat([shap_df, phys_shap_df], ignore_index=True)
            shap_df = shap_df.sort_values(
                ["TrainDays", "Date", "Target", "Model", "Feature"], kind="stable"
            ).reset_index(drop=True)
    phys_df = compute_physical_metrics(df_h, config=effective_config, verbose=effective_verbose)
    summary_df = build_optuna_summary_table(metrics_df, phys_df=phys_df, config=effective_config)
    verbose_print(f"Optuna summary table built: rows={len(summary_df)}.", verbose=effective_verbose, level=1)

    if export_excel:
        verbose_print(
            f"Exporting Optuna CSV results to {results_output_dir.resolve()}.",
            verbose=effective_verbose,
            level=1,
        )
        export_optuna_results(
            metrics_df=metrics_df,
            summary_df=summary_df,
            trials_df=trials_df,
            output_dir=results_output_dir,
            phys_df=phys_df,
            preds_df=preds_df,
            df_h=df_h,
            config=effective_config,
            excel_name_suffix=resolved_export_suffix,
            excel_timestamp=export_timestamp,
            shap_df=shap_df,
        )
    verbose_print("Optuna pipeline finished successfully.", verbose=effective_verbose, level=1)

    return {
        "config": effective_config,
        "optuna_config": optuna_config,
        "iri_config": iri_config,
        "df_raw": df_raw,
        "df_daily": df_daily,
        "df_h": df_h,
        "savgol_diagnostics_df": savgol_diagnostics_df,
        "savgol_metadata": savgol_metadata,
        "metrics_df": metrics_df,
        "trials_df": trials_df,
        "preds_df": preds_df,
        "phys_df": phys_df,
        "summary_df": summary_df,
        "shap_df": shap_df,
        "feature_pool": get_feature_pool(df_h),
        "station_metadata": station_metadata,
        "source_paths": source_paths,
        "export_dir": str(results_output_dir.resolve()),
    }


def _normalize_station_codes(
    base_dir: str | Path,
    datasets_dir: str,
    station_codes: Iterable[str] | None,
) -> list[str]:
    available = list_available_stations(base_dir=base_dir, datasets_dir=datasets_dir)
    available_codes = available["station_code"].tolist()
    if station_codes is None:
        return available_codes

    normalized = []
    seen = set()
    missing = []
    for station_code in station_codes:
        code = station_code.strip().upper()
        if code in seen:
            continue
        seen.add(code)
        if code not in available_codes:
            missing.append(code)
            continue
        normalized.append(code)
    if missing:
        raise ValueError(
            f"Unknown station codes: {', '.join(missing)}. "
            f"Available codes: {', '.join(sorted(available_codes))}."
        )
    return normalized


def _run_optuna_pipeline_station_task(
    station_code: str,
    base_dir: str | Path,
    datasets_dir: str,
    config: PipelineConfig,
    optuna_config: OptunaConfig,
    export_excel: bool,
    compute_shap: bool,
    verbose: int | None,
    excel_name_suffix: str | None,
) -> tuple[str, dict[str, pd.DataFrame | list[str] | PipelineConfig | OptunaConfig | IRIConfig]]:
    try:
        result = run_optuna_pipeline(
            base_dir=base_dir,
            datasets_dir=datasets_dir,
            station_code=station_code,
            config=config,
            optuna_config=optuna_config,
            export_excel=export_excel,
            compute_shap=compute_shap,
            verbose=verbose,
            excel_name_suffix=excel_name_suffix,
        )
    except Exception as exc:
        raise RuntimeError(f"Station {station_code} failed: {exc}") from exc
    return station_code, result


def run_optuna_pipeline_for_stations(
    base_dir: str | Path = ".",
    datasets_dir: str = "datasets",
    station_codes: Iterable[str] | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG,
    max_workers: int | None = None,
    export_excel: bool = False,
    compute_shap: bool = True,
    verbose: int | None = None,
    excel_name_suffix: str | None = None,
) -> dict[str, dict[str, pd.DataFrame | list[str] | PipelineConfig | OptunaConfig | IRIConfig]]:
    """Run the Optuna pipeline for multiple stations sequentially."""

    resolved_codes = _normalize_station_codes(
        base_dir=base_dir,
        datasets_dir=datasets_dir,
        station_codes=station_codes,
    )
    if not resolved_codes:
        return {}

    effective_verbose = resolve_verbose(config=config, verbose=verbose)
    verbose_print(
        f"Running Optuna pipeline for {len(resolved_codes)} station(s) sequentially.",
        verbose=effective_verbose,
        level=1,
    )

    results = {}
    for station_code in resolved_codes:
        _, result = _run_optuna_pipeline_station_task(
            station_code,
            base_dir,
            datasets_dir,
            config,
            optuna_config,
            export_excel,
            compute_shap,
            verbose,
            _resolve_station_excel_name_suffix(
                excel_name_suffix,
                station_code=station_code,
                ensure_unique=export_excel,
            ),
        )
        results[station_code] = result
    return results
