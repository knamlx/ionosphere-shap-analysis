from __future__ import annotations
 
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any
 
from ionosphere_pipeline import (
    DEFAULT_CONFIG,
    DEFAULT_IRI_CONFIG,
    DEFAULT_OPTUNA_CONFIG,
    IRIConfig,
    OptunaConfig,
    PipelineConfig,
    run_optuna_pipeline,
    run_pipeline,
    resolve_dataset_source_paths,
    load_giro_dataset,
    build_dataset_frame,
    add_iri_baseline,
    add_analytic_baseline,
    resolve_iri_config_from_metadata,
    resolve_results_output_dir,
    build_excel_timestamp_token,
    export_dataframe_csv,
    compute_physical_shap_records_for_optuna,
    OPTUNA_SHAP_RESULT_COLUMNS,
)
 
 
def _parse_datetime(value: Any) -> Any:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return value
 
 
def _parse_seasons(raw_seasons: Any) -> Any:
    if raw_seasons is None:
        return None
    parsed: dict[str, tuple[datetime, datetime]] = {}
    for season_name, bounds in dict(raw_seasons).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"Season {season_name!r} must contain exactly two ISO datetime strings: [start, end]."
            )
        parsed[str(season_name)] = (_parse_datetime(bounds[0]), _parse_datetime(bounds[1]))
    return parsed
 
 
def _build_pipeline_config(raw_config: dict[str, Any] | None) -> PipelineConfig:
    if not raw_config:
        return DEFAULT_CONFIG
    config_data = dict(raw_config)
    if "window_list" in config_data:
        config_data["window_list"] = tuple(int(value) for value in config_data["window_list"])
    if "targets" in config_data:
        config_data["targets"] = tuple(str(value) for value in config_data["targets"])
    if "savgol_target_columns" in config_data:
        config_data["savgol_target_columns"] = tuple(str(value) for value in config_data["savgol_target_columns"])
    if "shap_ml_features" in config_data:
        config_data["shap_ml_features"] = tuple(str(value) for value in config_data["shap_ml_features"])
    if "ml_date_start" in config_data:
        config_data["ml_date_start"] = config_data["ml_date_start"]
    if "ml_date_end" in config_data:
        config_data["ml_date_end"] = config_data["ml_date_end"]
    if "seasons" in config_data:
        config_data["seasons"] = _parse_seasons(config_data["seasons"])
    return PipelineConfig(**config_data)
 
 
def _build_optuna_config(raw_config: dict[str, Any] | None) -> OptunaConfig:
    if not raw_config:
        return DEFAULT_OPTUNA_CONFIG
    config_data = dict(raw_config)
    if "models" in config_data:
        config_data["models"] = tuple(str(value) for value in config_data["models"])
    return OptunaConfig(**config_data)
 
 
def _build_iri_config(raw_config: dict[str, Any] | None) -> IRIConfig:
    if not raw_config:
        return DEFAULT_IRI_CONFIG
    return IRIConfig(**dict(raw_config))
 
 
def load_runner_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Runner config file must contain a top-level JSON object.")
    return config
 
 
def run_physical_shap_only(
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    """Run only physical SHAP (IRI + Аналит.) without ML training.
 
    Loads the dataset, builds df_h, computes physical SHAP for all windows,
    and appends the results to the existing per-window shap CSV files.
 
    Activated when "models" in the runner JSON contains only physical model names,
    e.g. "models": ["IRI", "Аналит."]
    """
    import pandas as pd
 
    base_dir     = raw_config.get("base_dir", ".")
    datasets_dir = raw_config.get("datasets_dir", "datasets")
    station_code = raw_config.get("station_code")
    verbose      = raw_config.get("verbose", 1)
    pipeline_config = _build_pipeline_config(raw_config.get("pipeline_config"))
    iri_config_raw  = _build_iri_config(raw_config.get("iri_config"))
 
    print(f"[physical-only] station={station_code}, windows={pipeline_config.window_list}")
 
    # load data 
    source_paths, dataset_metadata = resolve_dataset_source_paths(
        base_dir=base_dir,
        datasets_dir=datasets_dir,
        station_code=station_code,
    )
    df_raw = load_giro_dataset(source_paths, min_cs=pipeline_config.min_cs, verbose=verbose)
    station_metadata = dict(df_raw.attrs.get("station_metadata", {}))
    for key, value in dataset_metadata.items():
        station_metadata.setdefault(key, value)
 
    iri_config = resolve_iri_config_from_metadata(
        station_metadata,
        iri_config=iri_config_raw,
        verbose=verbose,
    )
 
    df_h = build_dataset_frame(df_raw, config=pipeline_config, verbose=verbose)
    df_h = add_iri_baseline(df_h, config=pipeline_config, iri_config=iri_config)
    df_h = add_analytic_baseline(df_h, config=pipeline_config)
 
    #  compute physical SHAP 
    phys_shap_df = compute_physical_shap_records_for_optuna(
        df_h, config=pipeline_config, verbose=verbose
    )
 
    if phys_shap_df.empty:
        print("[physical-only] No physical SHAP records generated.")
        return {"phys_shap_df": phys_shap_df, "station_metadata": station_metadata}
 
    #  append to existing per-window shap CSV files 
    results_dir = resolve_results_output_dir(base_dir, station_metadata.get("station_code") or station_code)
    timestamp   = build_excel_timestamp_token()
    scode       = station_metadata.get("station_code") or station_code or "station"
 
    for train_days in pipeline_config.window_list:
        window_df = phys_shap_df[phys_shap_df["TrainDays"] == train_days].copy()
        if window_df.empty:
            continue
 
        # Find existing shap CSV for this window
        pattern = f"*window_{train_days}d*shap*.csv"
        existing = list(results_dir.glob(pattern))
 
        if existing:
            existing_path = existing[0]
            base_df = pd.read_csv(existing_path)
            # Remove old physical rows if any (re-compute)
            base_df = base_df[~base_df["Model"].isin(["IRI", "Аналит."])]
            merged  = pd.concat([base_df, window_df], ignore_index=True)
            merged  = merged.sort_values(
                ["TrainDays", "Date", "Target", "Model", "Feature"], kind="stable"
            ).reset_index(drop=True)
            merged.to_csv(existing_path, index=False)
            print(f"[physical-only] Appended to {existing_path.name}: +{len(window_df)} rows")
        else:
            # No existing file — write new one
            out_name = f"{scode}_window_{train_days}d_shap_physical_{timestamp}.csv"
            out_path = results_dir / out_name
            export_dataframe_csv(window_df, out_path)
            print(f"[physical-only] Written new file: {out_name}")
 
    return {
        "phys_shap_df": phys_shap_df,
        "station_metadata": station_metadata,
        "export_dir": str(results_dir.resolve()),
    }
 
 
def run_from_config(config_path: str | Path) -> dict[str, Any]:
    raw_config = load_runner_config(config_path)
    pipeline_kind = str(raw_config.get("pipeline", "optuna")).strip().lower()
    base_dir = raw_config.get("base_dir", ".")
    datasets_dir = raw_config.get("datasets_dir", "datasets")
    station_code = raw_config.get("station_code")
    export_excel = bool(raw_config.get("export_excel", True))
    compute_shap = bool(raw_config.get("compute_shap", True))
    compute_physical_shap = bool(raw_config.get("compute_physical_shap", True))
    verbose = raw_config.get("verbose")
    excel_name_suffix = raw_config.get("excel_name_suffix")
 
    #  model selection 
    # "models" in runner JSON lets you restrict which models run.
    # If only physical models are listed → skip ML entirely (fast path).
    PHYSICAL_MODELS = {"IRI", "Аналит."}
    runner_models = raw_config.get("models")  # e.g. ["IRI", "Аналит."] or None
 
    if runner_models is not None:
        runner_models_set = set(str(m) for m in runner_models)
        if runner_models_set <= PHYSICAL_MODELS:
            # Physical-only mode — skip Optuna ML training entirely
            print(f"[runner] models={runner_models} → physical-only mode")
            return run_physical_shap_only(raw_config)
        # Otherwise restrict ML models in optuna_config
        optuna_raw = dict(raw_config.get("optuna_config") or {})
        ml_only = [m for m in runner_models if m not in PHYSICAL_MODELS]
        if ml_only:
            optuna_raw["models"] = ml_only
            raw_config = dict(raw_config)
            raw_config["optuna_config"] = optuna_raw
 
    #  disable physical SHAP if requested 
    if compute_shap and not compute_physical_shap:
        import ionosphere_pipeline as _pip
        import pandas as _pd
        if hasattr(_pip, "compute_physical_shap_records_for_optuna"):
            _pip.compute_physical_shap_records_for_optuna = (
                lambda *a, **kw: _pd.DataFrame(columns=_pip.OPTUNA_SHAP_RESULT_COLUMNS)
            )
 
    pipeline_config = _build_pipeline_config(raw_config.get("pipeline_config"))
    iri_config = _build_iri_config(raw_config.get("iri_config"))
 
    if pipeline_kind == "standard":
        return run_pipeline(
            base_dir=base_dir,
            datasets_dir=datasets_dir,
            station_code=station_code,
            config=pipeline_config,
            iri_config=iri_config,
            export_excel=export_excel,
            compute_shap=compute_shap,
            verbose=verbose,
            excel_name_suffix=excel_name_suffix,
        )
 
    if pipeline_kind == "optuna":
        optuna_config = _build_optuna_config(raw_config.get("optuna_config"))
        return run_optuna_pipeline(
            base_dir=base_dir,
            datasets_dir=datasets_dir,
            station_code=station_code,
            config=pipeline_config,
            optuna_config=optuna_config,
            export_excel=export_excel,
            compute_shap=compute_shap,
            verbose=verbose,
            excel_name_suffix=excel_name_suffix,
        )
 
    raise ValueError("pipeline must be either 'standard' or 'optuna'.")
 
 
def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": asdict(result["config"]) if "config" in result else {},
        "station_metadata": result.get("station_metadata", {}),
        "savgol_metadata": result.get("savgol_metadata", {}),
    }
    if "export_dir" in result:
        summary["export_dir"] = result["export_dir"]
    for frame_name in ("metrics_df", "trials_df", "preds_df", "fi_df", "phys_df", "summary_df", "shap_df", "phys_shap_df"):
        frame = result.get(frame_name)
        if frame is not None:
            summary[f"{frame_name}_rows"] = int(len(frame))
    return summary
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ionosphere ML pipeline from a JSON config file.")
    parser.add_argument("config", help="Path to runner JSON config.")
    parser.add_argument(
        "--summary-json",
        help="Optional path to save a short JSON summary of the run.",
    )
    args = parser.parse_args()
 
    result = run_from_config(args.config)
    summary = build_summary(result)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
 
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 